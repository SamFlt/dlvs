import faulthandler
from functools import partial
from operator import add
from generator import MultiGeneratorHandler, SceneSet
import numpy as np
from torch.utils.data import Dataset

from geometry import batch_axis_angle_to_rotation_matrix, batch_rotation_matrix_to_axis_angle
import torch

import matplotlib.pyplot as plt
import cv2
from utils.datasets import *

class AEDataset(Dataset):
    def __init__(self, batches_per_epoch, batch_size, num_workers, generator_parameters):
        super(AEDataset, self).__init__()
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.generator_parameters = generator_parameters
        self.scenes = SceneSet(generator_parameters['image_paths'])
        self.num_workers = num_workers
        if num_workers > 0:
            ds = [generator_parameters.copy() for _ in range(num_workers)]
            for i in range(num_workers):
                ds[i]['base_seed'] = generator_parameters['base_seed'] * (i + 1) + 13
            
            self.generator_list = [MultiGeneratorHandler(batch_size//2, self.scenes, d) for d in ds]
        else:
            self.generator = MultiGeneratorHandler(batch_size//2, self.scenes, self.generator_parameters.copy())
    def preprocess(self, images):
        return (images - 127.5) / 127.5

    def __getitem__(self, item):
        res = self.generator.new_vs_examples(self.batch_size//2, 224, 224, 0, False, False)
        images = np.empty((self.batch_size, 1, 224, 224))
        for i, sample in enumerate(res):
            gray = cv2.cvtColor(np.array(sample.trajectory_images()[0], copy=False), cv2.COLOR_RGB2GRAY)
            images[i * 2] = np.expand_dims(gray, 0)
            gray = cv2.cvtColor(np.array(sample.desired_image(), copy=False), cv2.COLOR_RGB2GRAY)
            images[i * 2 + 1] = np.expand_dims(gray, 0)
        images = torch.from_numpy(self.preprocess(images))
        return images

    def get_worker_init_fn(self):
        def init_fn(worker_id):
            if self.num_workers > 0:
                worker_info = torch.utils.data.get_worker_info()
                dataset = worker_info.dataset
                dataset.generator = dataset.generator_list[worker_id]
        return init_fn
    def on_epoch_end(self, epoch):
        if self.num_workers > 0:
            for i in range(len(self.generator_list)):
                g = self.generator_parameters.copy()
                g['base_seed'] = g['base_seed'] + epoch * len(self.generator_list) * 50000 + (i + 1) * 50000
                self.generator_list[i] = MultiGeneratorHandler(self.batch_size//2, self.scenes, g)
            else:
                g = self.generator_parameters.copy()
                g['base_seed'] = g['base_seed'] + epoch * 50000
                self.generator = MultiGeneratorHandler(self.batch_size//2, self.scenes, g)

    def __len__(self):
        return self.batches_per_epoch

class StoredAEDataset(Dataset):
    def __init__(self, batches_per_epoch, batch_size, num_workers, generator_parameters):
        super(StoredAEDataset, self).__init__()
        assert num_workers == 0
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.generator_parameters = generator_parameters
        self.scenes = SceneSet(generator_parameters['image_paths'])
        self.num_workers = num_workers
        self.data = np.empty((batches_per_epoch * batch_size, 1, 224, 224), dtype=np.uint8)
        self.generator = MultiGeneratorHandler(batch_size//2, self.scenes, self.generator_parameters)
        for i in range(self.batches_per_epoch):
            res = self.generator.new_vs_examples(self.batch_size//2, 224, 224, 0, False, False)
            index_batch = self.batch_size * i
            for j, sample in enumerate(res):
                gray = cv2.cvtColor(np.array(sample.trajectory_images()[0], copy=False), cv2.COLOR_RGB2GRAY)
                self.data[index_batch + j * 2] = np.expand_dims(gray, 0)
                
                gray = cv2.cvtColor(np.array(sample.desired_image(), copy=False), cv2.COLOR_RGB2GRAY)
                self.data[index_batch + j * 2 + 1] = np.expand_dims(gray, 0)
            
        self.indices = np.arange(self.batch_size * self.batches_per_epoch)
        np.random.shuffle(self.indices)
    

    def __getitem__(self, item):
        d = self.data[item * self.batch_size:(item + 1) * self.batch_size]
        d = torch.from_numpy(d)
        return d
    def get_worker_init_fn(self):
        def init_fn(worker_id):
            pass
        return init_fn
    def on_epoch_end(self, epoch):
        np.random.shuffle(self.indices)
    def __len__(self):
        return self.batches_per_epoch


class StoredLookAtAEDataset(Dataset):
    def __init__(self, batches_per_epoch, batch_size, num_workers, generator_parameters, look_at_parameters, rz_max_deg=180.0, augmentation_factor=0.0, output_depth=False, use_gaussian_blur=False):
        super(StoredLookAtAEDataset, self).__init__()
        assert num_workers == 0
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.generator_parameters = generator_parameters
        self.scenes = SceneSet(generator_parameters['image_paths'])
        scene_count = len(generator_parameters['image_paths'])
        self.scene_indices = np.random.choice(scene_count, self.batch_size * self.batches_per_epoch, replace=True)
        self.num_workers = num_workers
        self.look_at_parameters = look_at_parameters
        self.num_samples = batches_per_epoch * batch_size
        self.rz_max = np.radians(rz_max_deg)
        

        h, w = self.generator_parameters['h'], self.generator_parameters['w']
        self.data = np.ones((self.num_samples, 1, h, w), dtype=np.uint8)
        self.depth = np.empty((self.num_samples, h, w), dtype=np.float) if output_depth else None
        self.generator = MultiGeneratorHandler(batch_size, self.scenes, self.generator_parameters)
        self.la_points = generate_look_at_positions(self.num_samples, self.look_at_parameters['look_at_half_ranges'])
        self.lf_points = generate_look_from_positions(self.num_samples, self.look_at_parameters['look_from_half_ranges'], self.look_at_parameters['center_Z'])
        self.la_Rs = compute_rotation_matrix_look_at(self.la_points, self.lf_points)
        self.Rz = add_optical_axis_rotations(self.la_Rs, self.rz_max)
        self.Rs = np.matmul(self.Rz, self.la_Rs)
        self.tus = batch_rotation_matrix_to_axis_angle(self.Rs)
        self.poses = np.concatenate((self.lf_points, self.tus), axis=-1)
        # self._debug()
        for i in range(self.batches_per_epoch):
            p = self.poses[i * self.batch_size: (i+1) * self.batch_size]
            si = self.scene_indices[i * self.batch_size: (i+1) * self.batch_size]
            res = self.generator.image_at_poses(self.batch_size, h, w, augmentation_factor, p, si, output_depth)
            images = [r.image() for r in res]

            for j, image in enumerate(images):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                self.data[i * self.batch_size + j] = np.expand_dims(gray, 0)
            if self.depth is not None:
                depths = [r.depth() for r in res]
                for j, depth in enumerate(depths):
                    self.depth[i * self.batch_size + j] = depth

            
        self.indices = np.arange(self.batch_size * self.batches_per_epoch)
        np.random.shuffle(self.indices)
        self.use_gaussian_blur = use_gaussian_blur
        if self.use_gaussian_blur:
            from utils.datasets import gaussian_blur_kernel
            ks = 5
            sigma = 2
            self.gaussian_kernel = torch.from_numpy(gaussian_blur_kernel(ks, sigma)).view(1, 1, ks, ks).float()


        # self._debug()
    
    def _debug(self):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs = [self.lf_points[:, i] for i in range(3)]
        ax.scatter(xs, ys, zs, c='r')
        fwd = batch_axis_angle_to_rotation_matrix(self.poses[:, 3:])[:, :, 2]
        u, v, w = [fwd[:, i] for i in range(3)]
        ax.quiver(xs,ys,zs, u, v, w, length=0.1)
        xs, ys, zs = [self.la_points[:, i] for i in range(3)]
        ax.scatter(xs, ys, zs, c='b')
        
        for i in range(self.num_samples):
            xs = self.la_points[i, 0], self.lf_points[i, 0]
            ys = self.la_points[i, 1], self.lf_points[i, 1]
            zs = self.la_points[i, 2], self.lf_points[i, 2]
            ax.plot(xs,ys,zs)
            

        
        # xs, ys, zs = [self.poses[:, i] for i in range(3)]
        # ax.scatter(xs, ys, zs, c='g')
        

        plt.show()
        plt.close()
        plt.figure()
        plt.imshow(self.data[0, 0])
        plt.show()

    def __getitem__(self, item):
        chosen_indices = self.indices[item * self.batch_size:(item + 1) * self.batch_size]
        d = self.data[chosen_indices]
        d = torch.from_numpy(d)
        if self.use_gaussian_blur:
            d = torch.nn.functional.conv2d(d.float(), self.gaussian_kernel, padding=2)
        p = self.poses[chosen_indices]
        p = torch.from_numpy(p)
        si = self.scene_indices[chosen_indices]
        si = torch.from_numpy(si)
        depths = None
        if self.depth is not None:
            depths = self.depth[chosen_indices]
        
        return d, p, si, depths
    def get_worker_init_fn(self):
        def init_fn(worker_id):
            pass
        return init_fn
    def on_epoch_end(self, epoch):
        np.random.shuffle(self.indices)
    def __len__(self):
        return self.batches_per_epoch

class StoredLookAtAEDatasetWithNearestNeighbors(Dataset):
    def __init__(self, batches_per_epoch, batch_size, select_k_nn, distance_metric, num_workers, generator_parameters, look_at_parameters, rz_max_deg=180.0, output_depth=False):
        super(StoredLookAtAEDatasetWithNearestNeighbors, self).__init__()
        assert num_workers == 0
        self.batch_size = batch_size
        self.select_k_nn = select_k_nn
        assert self.batch_size % (self.select_k_nn + 1) == 0, 'batch size must be dividable by nearest neighbours + 1'
        self.random_samples_per_batch = self.batch_size // (self.select_k_nn + 1)
        self.batches_per_epoch = batches_per_epoch
        self.generator_parameters = generator_parameters
        self.scenes = SceneSet(generator_parameters['image_paths'])
        scene_count = len(generator_parameters['image_paths'])
        self.scene_indices = np.random.choice(scene_count, self.batch_size * self.batches_per_epoch, replace=True)
        self.num_workers = num_workers
        self.look_at_parameters = look_at_parameters
        self.num_samples = batches_per_epoch * batch_size
        self.rz_max = np.radians(rz_max_deg)
        

        h, w = self.generator_parameters['h'], self.generator_parameters['w']
        self.data = np.ones((self.num_samples, 1, h, w), dtype=np.uint8)
        self.depth = np.empty((self.num_samples, h, w), dtype=np.float) if output_depth else None
        self.generator = MultiGeneratorHandler(batch_size, self.scenes, self.generator_parameters)
        self.la_points = generate_look_at_positions(self.num_samples, self.look_at_parameters['look_at_half_ranges'])
        self.lf_points = generate_look_from_positions(self.num_samples, self.look_at_parameters['look_from_half_ranges'], self.look_at_parameters['center_Z'])
        self.la_Rs = np.transpose(compute_rotation_matrix_look_at(self.la_points, self.lf_points), (0, 2 ,1))
        def Rz_rotation():
            theta = np.random.uniform(-self.rz_max, self.rz_max, size=len(self.la_Rs))
            c = np.cos(theta)
            s = np.sin(theta)
            Rz = np.zeros((len(self.la_Rs), 3, 3))
            Rz[:, 2, 2] = 1
            Rz[:, 0, 0] = c
            Rz[:, 1, 1] = c
            Rz[:, 0, 1] = -s
            Rz[:, 1, 0] = s
            return Rz
        # self.Rz = np.transpose(add_optical_axis_rotations(self.la_Rs, self.rz_max), (0, 2, 1))
        if self.rz_max > 0:
            self.Rz = np.transpose(Rz_rotation(), (0, 2, 1))
            self.Rs = np.matmul(self.la_Rs, self.Rz)
        else:
            self.Rs = self.la_Rs
        self.tus = batch_rotation_matrix_to_axis_angle(self.Rs)
        self.poses = np.concatenate((self.lf_points, self.tus), axis=-1)
        # self._debug()
        for i in range(self.batches_per_epoch):
            p = self.poses[i * self.batch_size: (i+1) * self.batch_size]
            si = self.scene_indices[i * self.batch_size: (i+1) * self.batch_size]
            res = self.generator.image_at_poses(self.batch_size, h, w, 0.0, p, si, output_depth)
            images = [r.image() for r in res]

            for j, image in enumerate(images):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                self.data[i * self.batch_size + j] = np.expand_dims(gray, 0)
            if self.depth is not None:
                depths = [r.depth() for r in res]
                for j, depth in enumerate(depths):
                    self.depth[i * self.batch_size + j] = depth
            

            
        self.indices = np.random.choice(self.num_samples, size=self.random_samples_per_batch * self.batches_per_epoch, replace=False)
        #np.arange(self.random_samples_per_batch * self.batches_per_epoch)
        #np.random.shuffle(self.indices)
        self.distance_metric = distance_metric
        if self.distance_metric == 'se3':
            self.t_dist_mean, self.r_dist_mean = self.get_scales(min(10000, self.num_samples))
        elif self.distance_metric == 'reprojection_error':
            xx = np.linspace(-0.4, 0.4, num=10)
            yy = np.linspace(-0.4, 0.4, num=10)
            xx, yy = np.meshgrid(xx, yy, indexing='ij')
            self.points3d = np.zeros((10 * 10, 3))
            self.points3d[:, 0] = xx.reshape(-1)
            self.points3d[:, 1] = yy.reshape(-1)

    def get_scales(self, n):
        random_samples = np.random.choice(self.num_samples, size=n, replace=False)

        p = self.poses[random_samples]
        t_dist, r_dist = get_se3_dist_matrices(p)
        triu = np.triu_indices(n, k=1)
        t_mean = np.mean(t_dist[triu])
        r_mean = np.mean(r_dist[triu])
        return t_mean, r_mean

    def __getitem__(self, item):
        random_sample_indices = self.indices[item * self.random_samples_per_batch:(item + 1) * self.random_samples_per_batch]

        p = self.poses[random_sample_indices]
        knn_indices = self.get_nearest_neighbours(p, random_sample_indices)
        assert knn_indices.shape == (self.random_samples_per_batch, self.select_k_nn)
        chosen_indices = np.concatenate((random_sample_indices[:, None], knn_indices), axis=-1)
        chosen_indices = np.reshape(chosen_indices, self.random_samples_per_batch * (self.select_k_nn + 1))
        chosen_indices = np.unique(chosen_indices) # Remove duplicate values => They bring no additional information for training
        p = self.poses[chosen_indices]
        p = torch.from_numpy(p)

        si = self.scene_indices[chosen_indices]
        d = self.data[chosen_indices]
        d = torch.from_numpy(d)
        
        si = torch.from_numpy(si)
        depths = None
        if self.depth is not None:
            depths = self.depth[chosen_indices]
        
        return d, p, si, depths
    def get_nearest_neighbours(self, p, pi, n=1000):
        #Sample random poses with which to compare p in order to select the k nearest neighbours of each p
        kr = min(self.num_samples - len(pi), n)
        probas = np.ones(len(self.poses)) / (len(self.poses) - len(pi)) # Uniform choice distribution, without indices in pi
        for index in pi:
            probas[index] = 0.0
        random_pose_indices = np.random.choice(len(self.poses), size=kr, replace=False, p=probas)
        p2 = self.poses[random_pose_indices]

        # Compute distance matrices
        if self.distance_metric == 'se3':
            t_dist, r_dist = get_se3_dist_matrices_compare(p, p2)
            dist = t_dist / self.t_dist_mean + r_dist / self.r_dist_mean
            assert dist.shape == (len(p), kr)
        elif self.distance_metric == 'reprojection_error':
            dist = compute_reprojection_error(self.points3d, p, p2)
        # For each  pose in p, get indices of nearest poses
        dist_sort_indices = np.argsort(dist, axis=-1) # Get indices that would sort se3 dist matrix, per pose in p 
        dist_knn = dist_sort_indices[:, :self.select_k_nn] # K nearest neighbours indices in randomly sampled batch
        dist_knn = random_pose_indices[dist_knn] # Select true pose indices
        return dist_knn

    def get_worker_init_fn(self):
        def init_fn(worker_id):
            pass
        return init_fn
    def on_epoch_end(self, epoch):
        self.indices = np.random.choice(self.num_samples, size=self.random_samples_per_batch * self.batches_per_epoch, replace=False)
    def __len__(self):
        return self.batches_per_epoch


def get_neighbours(i, poses, k, distance_metric, t_dist_mean, r_dist_mean, points3d):
    pi = poses[i:i+1]
    # Compute distance matrices
    if distance_metric == 'se3':
        t_dist, r_dist = get_se3_dist_matrices_compare(pi, poses)
        dist = t_dist / t_dist_mean + r_dist / r_dist_mean
        assert dist.shape == (1, len(poses))
    elif distance_metric == 'reprojection_error':
        dist = compute_reprojection_error(points3d, pi, poses)
    # For each  pose in p, get indices of nearest poses
    dist_sort_indices = np.argsort(dist, axis=-1)[0] # Get indices that would sort se3 dist matrix, per pose in p 
    indices = []
    j = 0
    while len(indices) < k:
        v = dist_sort_indices[j]
        if  v != i: # Do not take the sample as its own neigbhbour
            indices.append(v)
        j += 1
    return np.array(indices)

class StoredLookAtAEDatasetWithNearestNeighborsV2(StoredLookAtAEDatasetWithNearestNeighbors):
    def __init__(self, batches_per_epoch, batch_size, store_k_nn, select_k_nn,
                       distance_metric, num_workers, generator_parameters, look_at_parameters,
                       rz_max_deg=180.0, output_depth=False):
        super().__init__(batches_per_epoch, batch_size, select_k_nn,
                        distance_metric, num_workers, generator_parameters,
                        look_at_parameters, rz_max_deg, output_depth)
        self.store_k_nn = store_k_nn
        print(f'Store = {self.store_k_nn}, Select = {self.select_k_nn}')
        self.points3d = None
        
        print(f'Computing and storing first {self.store_k_nn} nearest neighbours for each sample')
        self.k_indices = self.compute_first_k_nearest_neighbours()

    
    def compute_first_k_nearest_neighbours(self):
        res = np.empty((self.num_samples, self.store_k_nn), dtype=np.int)
        import multiprocessing
        from tqdm import tqdm
        with multiprocessing.Pool(processes=4) as pool:
            progress_bar = tqdm(total=self.num_samples)
            fn = partial(get_neighbours, poses=self.poses, k=self.store_k_nn, distance_metric=self.distance_metric,
                        t_dist_mean=self.t_dist_mean, r_dist_mean=self.r_dist_mean, points3d=self.points3d)
            outputs = tqdm(pool.imap(fn, range(self.num_samples)), total=self.num_samples)
            res = np.array(list(outputs))
        
        return res

    def __getitem__(self, item):
        random_sample_indices = self.indices[item * self.random_samples_per_batch:(item + 1) * self.random_samples_per_batch]

        p = self.poses[random_sample_indices]
        nearest_neighbour_indices = self.k_indices[random_sample_indices]
        knn_indices = np.zeros((self.random_samples_per_batch, self.select_k_nn), dtype=np.int)
        for i in range(self.random_samples_per_batch):
            chosen_ks = np.random.choice(self.store_k_nn, size=self.select_k_nn, replace=False)
            knn_indices[i] = nearest_neighbour_indices[i, chosen_ks]
        
        assert knn_indices.shape == (self.random_samples_per_batch, self.select_k_nn)

        chosen_indices = np.concatenate((random_sample_indices[:, None], knn_indices), axis=-1)
        chosen_indices = np.reshape(chosen_indices, self.random_samples_per_batch * (self.select_k_nn + 1))
        
        p = self.poses[chosen_indices]
        p = torch.from_numpy(p)
        si = self.scene_indices[chosen_indices]
        d = self.data[chosen_indices]
        d = torch.from_numpy(d)
        si = torch.from_numpy(si)
        depths = None
        if self.depth is not None:
            depths = self.depth[chosen_indices]
        return d, p, si, depths



class TranslationDataset(Dataset):
    def __init__(self, batches_per_epoch, batch_size, num_workers, generator_parameters, translation_ranges):
        super(TranslationDataset, self).__init__()
        assert num_workers == 0
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.generator_parameters = generator_parameters
        self.scenes = SceneSet(generator_parameters['image_paths'])
        scene_count = len(generator_parameters['image_paths'])
        self.scene_indices = np.random.choice(scene_count, self.batch_size * self.batches_per_epoch, replace=True)
        self.num_workers = num_workers
        self.num_samples = batches_per_epoch * batch_size
        self.translation_ranges = translation_ranges
        self.Z_center = -generator_parameters['base_camera_height']
        

        h, w = self.generator_parameters['h'], self.generator_parameters['w']
        self.data = np.ones((self.num_samples, 1, h, w), dtype=np.uint8)
        self.generator = MultiGeneratorHandler(batch_size, self.scenes, self.generator_parameters)
        self.translations = self.generate_positions(self.num_samples)
        self.poses = np.concatenate((self.translations, np.zeros_like(self.translations)), axis=-1)
        for i in range(self.batches_per_epoch):
            p = self.poses[i * self.batch_size: (i+1) * self.batch_size]
            si = self.scene_indices[i * self.batch_size: (i+1) * self.batch_size]
            res = self.generator.image_at_poses(self.batch_size, h, w, 0.0, p, si, False)
            images = [r.image() for r in res]

            for j, image in enumerate(images):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                self.data[i * self.batch_size + j] = np.expand_dims(gray, 0)
            

            
        self.indices = np.arange(self.batch_size * self.batches_per_epoch)
        np.random.shuffle(self.indices)



    def __getitem__(self, item):
        chosen_indices = self.indices[item * self.batch_size:(item + 1) * self.batch_size]
        d = self.data[chosen_indices]
        d = torch.from_numpy(d)
        p = self.poses[chosen_indices]
        p = torch.from_numpy(p)
        si = self.scene_indices[chosen_indices]
        si = torch.from_numpy(si)
        
        return d, p, si
    def get_worker_init_fn(self):
        def init_fn(worker_id):
            pass
        return init_fn
    def on_epoch_end(self, epoch):
        np.random.shuffle(self.indices)
    def __len__(self):
        return self.batches_per_epoch
    def generate_positions(self, num_samples):
        res = np.empty((num_samples, 3))
        base_Z = self.Z_center
        tr = self.translation_ranges
        for i in range(3):
            res[:, i] = np.random.uniform(-tr[i], tr[i], size=num_samples)

        res[:, 2] += base_Z
        return res

class FourDOFDataset(Dataset):
    def __init__(self, batches_per_epoch, batch_size, num_workers, generator_parameters, translation_ranges, max_rz_deg):
        super(FourDOFDataset, self).__init__()
        assert num_workers == 0
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.generator_parameters = generator_parameters
        self.scenes = SceneSet(generator_parameters['image_paths'])
        scene_count = len(generator_parameters['image_paths'])
        self.scene_indices = np.random.choice(scene_count, self.batch_size * self.batches_per_epoch, replace=True)
        self.num_workers = num_workers
        self.num_samples = batches_per_epoch * batch_size
        self.translation_ranges = translation_ranges
        self.rz_max = np.radians(max_rz_deg)
        self.Z_center = -generator_parameters['base_camera_height']

        h, w = self.generator_parameters['h'], self.generator_parameters['w']
        self.data = np.ones((self.num_samples, 1, h, w), dtype=np.uint8)
        self.generator = MultiGeneratorHandler(batch_size, self.scenes, self.generator_parameters)
        self.translations, self.rzs = self.generate_positions(self.num_samples)
        self.poses = np.concatenate((self.translations, np.zeros((self.num_samples, 2)), self.rzs), axis=-1)
        for i in range(self.batches_per_epoch):
            p = self.poses[i * self.batch_size: (i+1) * self.batch_size]
            si = self.scene_indices[i * self.batch_size: (i+1) * self.batch_size]
            res = self.generator.image_at_poses(self.batch_size, h, w, 0.0, p, si, False)
            images = [r.image() for r in res]

            for j, image in enumerate(images):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                self.data[i * self.batch_size + j] = np.expand_dims(gray, 0)
            

            
        self.indices = np.arange(self.batch_size * self.batches_per_epoch)
        np.random.shuffle(self.indices)



    def __getitem__(self, item):
        chosen_indices = self.indices[item * self.batch_size:(item + 1) * self.batch_size]
        d = self.data[chosen_indices]
        d = torch.from_numpy(d)
        p = self.poses[chosen_indices]
        p = torch.from_numpy(p)
        si = self.scene_indices[chosen_indices]
        si = torch.from_numpy(si)
        
        return d, p, si
    def get_worker_init_fn(self):
        def init_fn(worker_id):
            pass
        return init_fn
    def on_epoch_end(self, epoch):
        np.random.shuffle(self.indices)
    def __len__(self):
        return self.batches_per_epoch
    def generate_positions(self, num_samples):
        t = np.empty((num_samples, 3))
        base_Z = self.Z_center
        tr = self.translation_ranges
        for i in range(3):
            t[:, i] = np.random.uniform(-tr[i], tr[i], size=num_samples)
        t[:, 2] += base_Z

        rzr = self.rz_max
        rz = np.random.uniform(-rzr, rzr, size=(num_samples, 1))
        return t, rz


    
class StoredLookAtAEDatasetWithAugmentation(Dataset):
    def __init__(self, dataset: StoredLookAtAEDataset, augmentation_factor, augmentations_per_batch):
        super(StoredLookAtAEDatasetWithAugmentation, self).__init__()
        self.dataset = dataset
        self.augmentation_factor = augmentation_factor
        self.augmentations_per_batch = augmentations_per_batch
    
    def __getitem__(self, item):
        d, poses, scene_indices, _ = self.dataset[item]
        poses = poses.numpy()
        scene_indices = scene_indices.numpy()
        h, w = self.dataset.generator_parameters['h'], self.dataset.generator_parameters['w']
        bs = self.dataset.batch_size
        final_data = np.empty((bs, self.augmentations_per_batch + 1, 1, h, w))
        final_data[:, 0] = d
        for a in range(1, self.augmentations_per_batch + 1):
                
                res = self.dataset.generator.image_at_poses(bs, h, w, self.augmentation_factor, poses, scene_indices, False)
                images = [r.image() for r in res]

                for j, image in enumerate(images):
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    final_data[j, a] = np.expand_dims(gray, 0)
        return torch.from_numpy(final_data)



    def get_worker_init_fn(self):
        def init_fn(worker_id):
            pass
        return init_fn
    def on_epoch_end(self, epoch):
        self.dataset.on_epoch_end(epoch)
    def __len__(self):
        return self.dataset.batches_per_epoch