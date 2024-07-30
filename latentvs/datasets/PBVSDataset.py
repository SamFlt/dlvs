from generator import MultiGeneratorHandler, SceneSet
import numpy as np
from torch.utils.data import Dataset
import torch
from geometry import batch_homogeneous_inverse, batch_rotation_matrix_to_axis_angle, batch_to_homogeneous_transform_with_axis_angle, batch_rotation_matrix_to_axis_angle
import cv2
class PBVSDataset(Dataset):
    def __init__(self, output_depth, batches_per_epoch, batch_size, generator_parameters, num_workers, augmentation_factor=0.0, output_clean_images=True):
        super(PBVSDataset, self).__init__()
        print('Building PBVS dataset')
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.output_depth = output_depth
        self.output_clean_images = output_clean_images
        self.generator_parameters = generator_parameters
        self.scenes = SceneSet(generator_parameters['image_paths'])
        #self.generator = MultiGeneratorHandler(batch_size, generator_parameters)
        self.augmentation_factor = augmentation_factor
        self.num_workers = num_workers

        if self.num_workers > 0:
            ds = [generator_parameters.copy() for _ in range(num_workers)]
            for i in range(num_workers):
                ds[i]['base_seed'] = generator_parameters['base_seed'] * (i + 1) + 13

            self.generator_list = [MultiGeneratorHandler(batch_size, self.scenes, d) for d in ds]
        else:
            self.generator = MultiGeneratorHandler(batch_size, self.scenes, generator_parameters)


    def __getitem__(self, item):
        res = self.generator.new_vs_examples(self.batch_size, 224, 224, self.augmentation_factor, self.output_depth, self.output_clean_images)
        Is = np.empty((self.batch_size, 1, 224, 224))
        Ids = np.empty((self.batch_size, 1, 224, 224))
        I_inv_depths = None
        Id_inv_depths = None
        Is_clean = None
        Ids_clean = None
        if self.output_depth:
            I_inv_depths = np.empty((self.batch_size, 1, 224, 224))
            Id_inv_depths = np.empty((self.batch_size, 1, 224, 224))
        if self.output_clean_images:
            Is_clean = np.empty((self.batch_size, 1, 224, 224))
            Ids_clean = np.empty((self.batch_size, 1, 224, 224))
        velocities = np.empty((self.batch_size, 6))
        cs = np.array([s.trajectory_pose_vectors()[0] for s in res])
        cds = np.array([s.desired_pose_vector() for s in res])
        wTcs = batch_to_homogeneous_transform_with_axis_angle(cs[:, :3], cs[:, 3:])
        wTcds = batch_to_homogeneous_transform_with_axis_angle(cds[:, :3], cds[:, 3:])
        csTw = batch_homogeneous_inverse(wTcs)
        cdsTw = batch_homogeneous_inverse(wTcds)
        # cdsTcs = np.matmul(cdsTw, wTcs)
        # ts = cdsTcs[:, :3, 3]
        # tus = batch_rotation_matrix_to_axis_angle(cdsTcs[:, :3, :3])
        # dists_t = torch.from_numpy(np.linalg.norm(ts, axis=1).astype(np.float32))
        # dists_r = torch.from_numpy(np.linalg.norm(tus, axis=1).astype(np.float32))
        caTw = np.concatenate((csTw, cdsTw), axis=0)
        wTca = np.concatenate((wTcs, wTcds), axis=0)
        r_dist_matrix = np.empty((self.batch_size * 2, self.batch_size * 2), dtype=np.float32)
        t_dist_matrix = np.empty((self.batch_size * 2, self.batch_size * 2), dtype=np.float32)
        for i in range(self.batch_size * 2):
            caTci = np.matmul(caTw, wTca[i:i+1])
            ts = caTci[:, :3, 3]
            rs = batch_rotation_matrix_to_axis_angle(caTci[:, :3, :3])
            r_dist_matrix[i] = np.linalg.norm(rs, axis=-1)
            t_dist_matrix[i] = np.linalg.norm(ts, axis=-1)
        
        for i, sample in enumerate(res):
            I = np.array(sample.trajectory_images()[0], copy=False)
            Id = np.array(sample.desired_image(), copy=False)
            if self.output_depth:
                I_depth = np.array(sample.trajectory_depths()[0], copy=False)
                Id_depth = np.array(sample.desired_depth(), copy=False)
                I_inv_depths[i] = np.expand_dims(1.0 / I_depth, 0)
                Id_inv_depths[i] = np.expand_dims(1.0 / Id_depth, 0)
            if self.output_clean_images:
                Is_clean[i] = cv2.cvtColor(np.array(sample.clean_trajectory_images()[0], copy=False), cv2.COLOR_RGB2GRAY)
                Ids_clean[i] = cv2.cvtColor(np.array(sample.clean_desired_image(), copy=False), cv2.COLOR_RGB2GRAY)
            vc = np.array(sample.trajectory_velocities()[0], copy=False)
            I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
            Id = cv2.cvtColor(Id, cv2.COLOR_RGB2GRAY)
            Is[i] = np.expand_dims(I, 0)
            Ids[i] = np.expand_dims(Id, 0)
            velocities[i] = vc

        Is = torch.from_numpy(Is.astype(np.float32))
        Ids = torch.from_numpy(Ids.astype(np.float32))
        velocities = torch.from_numpy(velocities.astype(np.float32))
        if self.output_depth:
            I_inv_depths = torch.from_numpy(I_inv_depths.astype(np.float32))
            Id_inv_depths = torch.from_numpy(Id_inv_depths.astype(np.float32))
        if self.output_clean_images:
            Is_clean = torch.from_numpy(Is_clean.astype(np.float32))
            Ids_clean = torch.from_numpy(Ids_clean.astype(np.float32))
        t_dist_matrix = torch.from_numpy(t_dist_matrix)
        r_dist_matrix = torch.from_numpy(r_dist_matrix)
        return {
            'Is': Is,
            'Ids': Ids,
            'vc': velocities,
            'Is_clean': Is_clean,
            'Ids_clean': Ids_clean,
            'I_inv_depths': I_inv_depths,
            'Id_inv_depths': Id_inv_depths,
            't_dist_matrix': t_dist_matrix,
            'r_dist_matrix': r_dist_matrix
        }

    def get_worker_init_fn(self):
        def init_fn(worker_id):
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            dataset.generator = dataset.generator_list[worker_id]
        return init_fn
    def on_epoch_end(self, epoch):
        if self.num_workers > 0:
            for i in range(len(self.generator_list)):
                g = self.generator_parameters.copy()
                g['base_seed'] = g['base_seed'] + epoch * len(self.generator_list) * 50000 + (i + 1) * 50000
                self.generator_list[i] = MultiGeneratorHandler(self.batch_size, self.scenes, g)
        else:
            g = self.generator_parameters.copy()
            g['base_seed'] = g['base_seed'] + epoch * 5000
            self.generator = MultiGeneratorHandler(self.batch_size, self.scenes, g)
    def max_distances(self, sigma_distance=3.0):
        sigmas = np.max(self.generator_parameters['gaussian_sets_sigmas'], axis=0)
        sigmasn = sigmas * sigma_distance
        sigmasn_t, sigmasn_rot = sigmasn[:2], sigmasn[2:]
        dist_t = np.sqrt(sigmasn_t[0] ** 2 + sigmasn_t[0] ** 2 + sigmasn_t[1] ** 2)
        dist_r = np.radians(np.sqrt(sigmasn_rot[0] ** 2 + sigmasn_rot[0] ** 2 + sigmasn_rot[1] ** 2))
        return dist_t, dist_r

    def __len__(self):
        return self.batches_per_epoch

class StoredPBVSDataset(Dataset):
    def __init__(self, output_depth, batches_per_epoch, batch_size, generator_parameters, augmentation_factor=0.0, output_clean_images=True):
        super(StoredPBVSDataset, self).__init__()
        print('Building PBVS dataset')
        self.dataset = PBVSDataset(output_depth, batches_per_epoch, batch_size, generator_parameters, 0, augmentation_factor, output_clean_images)
        self.data = {}

        

        for i in range(len(self.dataset)):
            res = self.dataset[i]
            for k, d in res.items():
                if d is None:
                    self.data[k] = None
                elif k not in self.data:
                    self.data[k] = d
                else:
                    # print(k, d)
                    self.data[k] = np.concatenate((self.data[k], d), axis=0)

        self.indices = np.arange(self.dataset.batch_size * self.dataset.batches_per_epoch)
        np.random.shuffle(self.indices)


    def __getitem__(self, item):
        chosen_i = self.indices[item*self.dataset.batch_size:(item+1) * self.dataset.batch_size]
        out = {}
        for k, v in self.data.items():
            if self.data[k] is not None:
                out[k] = np.take(v, chosen_i, axis=0)
                print(k, out[k].shape)
            else:
                out[k] = None
        return out

    def get_worker_init_fn(self):
        def init_fn(worker_id):
            pass
        return init_fn
    def on_epoch_end(self, epoch):
        np.random.shuffle(self.indices)
    def __len__(self):
        return self.dataset.batches_per_epoch


class StoredPBVSDatasetV2(Dataset):
    def __init__(self, batches_per_epoch, batch_size, generator_parameters, augmentation_factor=0.0):
        super(StoredPBVSDatasetV2, self).__init__()
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.generator_parameters = generator_parameters
        self.scenes = SceneSet(generator_parameters['image_paths'])
        #self.generator = MultiGeneratorHandler(batch_size, generator_parameters)
        self.augmentation_factor = augmentation_factor
        self.generator = MultiGeneratorHandler(self.batch_size, self.scenes, generator_parameters)
        self.Is = np.empty((batches_per_epoch * batch_size, 1, 224, 224), dtype=np.uint8)
        self.Ids = np.empty((batches_per_epoch * batch_size, 1, 224, 224), dtype=np.uint8)
        self.wrcs = np.empty((batches_per_epoch * batch_size, 6))
        self.wrcds = np.empty((batches_per_epoch * batch_size, 6))
        
        self.wTcs = np.empty((batches_per_epoch * batch_size, 4, 4))
        self.wTcds = np.empty((batches_per_epoch * batch_size, 4, 4))
        self.velocities = np.empty((batches_per_epoch * batch_size, 6))

        for b in range(batches_per_epoch):
            # res = []
            # for _ in range(batch_size):
            res = self.generator.new_vs_examples(self.batch_size, 224, 224, self.augmentation_factor, False, False)

            cs = np.array([s.trajectory_pose_vectors()[0] for s in res])
            cds = np.array([s.desired_pose_vector() for s in res])
            self.wrcs[b*batch_size:(b+1) * batch_size] = cs
            self.wrcds[b*batch_size:(b+1) * batch_size] = cds
            
            wTcs = batch_to_homogeneous_transform_with_axis_angle(cs[:, :3], cs[:, 3:])
            wTcds = batch_to_homogeneous_transform_with_axis_angle(cds[:, :3], cds[:, 3:])
            self.wTcs[b*batch_size:(b+1) * batch_size] = wTcs
            self.wTcds[b*batch_size:(b+1) * batch_size] = wTcds
            
            for i, sample in enumerate(res):
                I = np.array(sample.trajectory_images()[0], copy=False)
                Id = np.array(sample.desired_image(), copy=False)
                
                vc = np.array(sample.trajectory_velocities()[0], copy=False)
                I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
                Id = cv2.cvtColor(Id, cv2.COLOR_RGB2GRAY)
                self.Is[b * batch_size + i, 0] = I
                self.Ids[b * batch_size + i, 0] = Id
                self.velocities[b * batch_size + i] = vc

        self.csTw = batch_homogeneous_inverse(self.wTcs)
        self.cdsTw = batch_homogeneous_inverse(self.wTcds)
        
        self.indices = np.arange(batch_size * batches_per_epoch)
        np.random.shuffle(self.indices)

    def __getitem__(self, item):
        chosen_indices = self.indices[item*self.batch_size:(item + 1) * self.batch_size]
        Is = self.Is[chosen_indices]
        Ids = self.Ids[chosen_indices]
        wrcs = self.wrcs[chosen_indices]
        wrcds = self.wrcds[chosen_indices]
        
        velocities = self.velocities[chosen_indices]
        wTcs = self.wTcs[chosen_indices]
        wTcds = self.wTcds[chosen_indices]
        
        csTw = self.csTw[chosen_indices]
        cdsTw = self.cdsTw[chosen_indices]
        caTw = np.concatenate((csTw, cdsTw), axis=0)
        wTca = np.concatenate((wTcs, wTcds), axis=0)
        r_dist_matrix = np.empty((self.batch_size * 2, self.batch_size * 2), dtype=np.float32)
        t_dist_matrix = np.empty((self.batch_size * 2, self.batch_size * 2), dtype=np.float32)
        for i in range(self.batch_size * 2):
            caTci = np.matmul(caTw, wTca[i:i+1])
            ts = caTci[:, :3, 3]
            rs = batch_rotation_matrix_to_axis_angle(caTci[:, :3, :3])
            r_dist_matrix[i] = np.linalg.norm(rs, axis=-1)
            t_dist_matrix[i] = np.linalg.norm(ts, axis=-1)
        # print(t_dist_matrix)


        return {
            'Is': torch.from_numpy(Is.astype(np.float32)),
            'Ids': torch.from_numpy(Ids.astype(np.float32)),
            'wrcs': torch.from_numpy(wrcs),
            'wrcds': torch.from_numpy(wrcds),
            'vc': torch.from_numpy(velocities.astype(np.float32)),
            't_dist_matrix': torch.from_numpy(t_dist_matrix),
            'r_dist_matrix': torch.from_numpy(r_dist_matrix)
        }
    
    def get_worker_init_fn(self):
        def init_fn(worker_id):
            pass
        return init_fn

    def on_epoch_end(self, epoch):
        np.random.shuffle(self.indices)
        
    def __len__(self):
        return self.batches_per_epoch
