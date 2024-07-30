import torch
from torch import nn
from enum import Enum
from pathlib import Path

from aevs.model.im_computable import permute_im_to_vec_rep_if_required_minimal_checks

from inference.interaction_matrix_mixer import *
from inference.utils import *
from inference.optimizers import *
from inference.methods import VSMethod
from utils.custom_typing import *
import yaml
from inference import io
from sklearn.neighbors import KDTree

from metric_learning.model.models import *

class InteractionMatrixFinder():
    '''
    Base class for methods that given a point in latent,
    examines its neighbours and interpolates their interaction matrices
    in order to find the interaction at the given point'''
    def __init__(self, datapoints, interaction_matrices):
        """

        Args:
            datapoints: the neighbours to examine when querying a point. An NxZ array
            interaction_matrices: the IMs associated to the datapoints. An NxZx6 array
        """        
        self.datapoints = datapoints
        self.interaction_matrices = interaction_matrices
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        '''
            For a given set of B points x (shape of BxZ), interpolate from their interaction matrix from the known datapoints
        '''
        raise NotImplementedError
    @staticmethod
    def make_from_yaml(node: yaml.Node) -> Callable[[torch.Tensor, torch.Tensor], 'InteractionMatrixFinder']:
        raise NotImplementedError
class NearestNeighbourAssigner(InteractionMatrixFinder):
    '''
    Class that sets the interaction matrix of a given point as the interaction matrix of its closest neighbour
    '''
    def __init__(self, datapoints, interaction_matrices):
        super().__init__(datapoints, interaction_matrices)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.sum((self.datapoints[None] - x.unsqueeze(1)) ** 2, dim=-1) # B x K
        nearest_index = torch.argmin(dists, dim=1)
        nearest_zp_im = self.interaction_matrices[nearest_index]
        return nearest_zp_im

    @staticmethod
    def make_from_yaml(node: yaml.Node) -> Callable[[torch.Tensor, torch.Tensor], 'InteractionMatrixFinder']:
        return lambda x, L: NearestNeighbourAssigner(x, L)


class KNNRegressor(InteractionMatrixFinder):
    '''
    Class that sets the interaction matrix of a given point as the distance-weighted average of its k nearest neighbours
    '''
    def __init__(self, k, datapoints, interaction_matrices, n_jobs):
        super().__init__(datapoints, interaction_matrices)
        from sklearn.neighbors import KNeighborsRegressor
        self.knn_model = KNeighborsRegressor(k, weights='distance', n_jobs=n_jobs, leaf_size=50)

        self.interaction_matrix_shape = self.interaction_matrices.size()
        self.knn_model = self.knn_model.fit(self.datapoints, self.interaction_matrices.view(self.interaction_matrix_shape[0], -1))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.knn_model.predict(x.cpu().numpy())
        prediction = torch.from_numpy(prediction).float().to(x.device)
        L = prediction.view(x.size(0), *self.interaction_matrix_shape[1:])
        return L
    @staticmethod
    def make_from_yaml(node: yaml.Node) -> Callable[[torch.Tensor, torch.Tensor], 'InteractionMatrixFinder']:
        return lambda x, L: KNNRegressor(node['k'], x.cpu(), L.cpu(), node['n_jobs'])


def make_interaction_matrix_finder_from_yaml(node: yaml.Node) -> Callable[[torch.Tensor, torch.Tensor], 'InteractionMatrixFinder']:
    res_dict = {
        'knn_regressor': KNNRegressor.make_from_yaml,
        '1nn': NearestNeighbourAssigner.make_from_yaml,
    }
    return res_dict[node['type']](node)

def compute_im(p):
    '''
    Compute the interaction matrix associated to a set of poses p
    p: Nx6 torch tensor, giving n poses of the form oTc, with o the fixed frame and c the camera
    returns the interaction matrices: Nx6x6
    '''
    def skew_sym(u):
        res = np.zeros((u.shape[0], 3, 3))
        res[:, 0, 1] = -u[:, 2]
        res[:, 0, 2] = u[:, 1]
        res[:, 1, 0] = u[:, 2]
        res[:, 1, 2] = -u[:, 0]
        res[:, 2, 0] = -u[:, 1]
        res[:, 2, 1] = u[:, 0]
        return res

    def sinc(t):
        res = np.sin(t) / t
        res[~np.isfinite(res)] = 1
        return res
    device = p.device
    p = p.cpu().numpy()
    res = np.zeros((p.shape[0], 6, 6))
    tu = p[:, 3:]
    R = batch_axis_angle_to_rotation_matrix(tu)
    res[:, :3, :3] = R
    theta = np.linalg.norm(tu, ord=2, axis=-1, keepdims=True)
    
    tx = theta.copy()
    tx[tx == 0] = 1

    u = tu / tx
    sk = skew_sym(u)

    I = np.repeat(np.eye(3).reshape((1, 3, 3)), u.shape[0], axis=0)
    v = sinc(theta) / (sinc(theta / 2) ** 2)
    Ltu = I - (theta[..., None] / 2) * sk + (1 - v[..., None]) * (sk ** 2)

    res[:, 3:, 3:] = Ltu

    return torch.from_numpy(res).to(device).float()

class PoseSampler():
    '''
    Abstract class that generates poses
    '''
    def __init__(self):
        pass

    def __call__(self, *args) -> torch.Tensor:
        '''
        Generate poses
        '''
        raise NotImplementedError
    def requires_data(self) -> bool:
        '''
        Whether the Pose generation process is dependent on outside data (such as a latent representation or another pose obtained during servoing)
        '''
        raise NotImplementedError
    def on_new_desired(self, zd):
        raise NotImplementedError
    def on_start_vs(self, z):
        raise NotImplementedError
    def on_end_vs(self, z):
        raise NotImplementedError
    @staticmethod
    def make_from_yaml(node: yaml.Node) -> 'PoseSampler':
        raise NotImplementedError
class GridPoseSampler(PoseSampler):
    def __init__(self, dofs, steps_per_dim, pose_center, ranges):
        super().__init__()
        assert dofs in (4, 6), 'Grid pose sampling is implemented only for 4 or 6 degrees of freedom'
        self.dofs = dofs
        self.steps_per_dim = steps_per_dim
        self.pose_center = pose_center
        self.ranges = ranges

    def __call__(self, *args: Any) -> torch.Tensor:
        r = self.ranges

        c = np.array([*self.pose_center, 0.0, 0.0, 0.0])
        wTc = batch_to_homogeneous_transform_with_axis_angle(c[None, :3], c[None, 3:])
        xs = torch.linspace(-r[0], r[0], steps=self.steps_per_dim) 
        ys = torch.linspace(-r[1], r[1], steps=self.steps_per_dim) 
        zs = torch.linspace(-r[2], r[2], steps=self.steps_per_dim)

        r3, r4, r5 = np.radians(r[3:])
        rzs = torch.linspace(-r5, r5, steps=self.steps_per_dim)
        v = []
        if self.dofs == 4:
            p = torch.meshgrid(xs, ys, zs, rzs)
            for i in range(len(p)):
                v.append(p[i].contiguous().view(-1, 1))
            v.insert(3, torch.zeros_like(v[0]))
            v.insert(4, torch.zeros_like(v[0]))
            v = torch.cat(v, dim=-1)
        elif self.dofs == 6:
            rxs = torch.linspace(-r3, r3, steps=self.steps_per_dim)
            rys = torch.linspace(-r4, r4, steps=self.steps_per_dim)
            p = torch.meshgrid(xs, ys, zs, rxs, rys, rzs)
            for i in range(len(p)):
                v.append(p[i].contiguous().view(-1, 1))
            v = torch.cat(v, dim=-1)
        cTs = batch_exponential_map(v.cpu().numpy(), dt=1.0)
        
        wTs = wTc @ cTs
        pp = batch_to_pose_vector(wTs)
        return torch.from_numpy(pp).float()
    def requires_data(self) -> bool:
        return False
    @staticmethod
    def make_from_yaml(node: yaml.Node) -> PoseSampler:
        return GridPoseSampler(node['dofs'], node['steps'], node['pose_center'], node['ranges'])
class GridPoseSamplerWithOversamplingCenter(GridPoseSampler):
    def __init__(self, dofs, steps_per_dim, pose_center, ranges, oversampling_count, oversampling_ranges):
        super().__init__(dofs, steps_per_dim, pose_center, ranges)
        self.oversampling_count = oversampling_count
        self.oversampling_ranges = np.array(oversampling_ranges)
        self.oversampling_ranges[3:] = np.radians(self.oversampling_ranges[3:])


    def __call__(self, *args: Any) -> torch.Tensor:
        pp = super().__call__(*args)
        v = np.empty((self.oversampling_count, 6))
        ra = self.oversampling_ranges
        for i in range(6):
            v[:, i] = np.random.uniform(-ra[i], ra[i], size=self.oversampling_count)
        cTs = batch_exponential_map(v, dt=1.0)
        c = np.array([*self.pose_center, 0.0, 0.0, 0.0])
        wTc = batch_to_homogeneous_transform_with_axis_angle(c[None, :3], c[None, 3:])
        wTs = wTc @ cTs
        pp_near = torch.from_numpy(batch_to_pose_vector(wTs)).float()
        pp_total = torch.cat((pp, pp_near), dim=0)


        return pp_total
    @staticmethod
    def make_from_yaml(node: yaml.Node) -> PoseSampler:
        return GridPoseSamplerWithOversamplingCenter(node['dofs'], node['steps'],
                                                     node['pose_center'], node['ranges'],
                                                     node['near_samples'], node['near_ranges'])
class UniformPoseSampler(PoseSampler):
    def __init__(self, count, pose_center, ranges):
        self.count = count
        self.pose_center = pose_center
        self.ranges = np.array(ranges)
        self.ranges[3:] = np.radians(self.ranges[3:])


    def __call__(self, *args: Any) -> torch.Tensor:
        v = np.empty((self.count, 6))
        ra = self.ranges
        for i in range(6):
            v[:, i] = np.random.uniform(-ra[i], ra[i], size=self.count)
        cTs = batch_exponential_map(v, dt=1.0)
        c = np.array([*self.pose_center, 0.0, 0.0, 0.0])
        wTc = batch_to_homogeneous_transform_with_axis_angle(c[None, :3], c[None, 3:])
        wTs = wTc @ cTs
        pp = torch.from_numpy(batch_to_pose_vector(wTs)).float()
        return pp
    @staticmethod
    def make_from_yaml(node: yaml.Node) -> PoseSampler:
        return UniformPoseSampler(node['count'], node['pose_center'], node['ranges'])

class DataBasedSampler(PoseSampler):
    def __init__(self, pose_center, initial_pose_count, random_sample_count):
        self.pose_center = pose_center
        self.initial_pose_count = initial_pose_count
        self.random_sample_count = random_sample_count

        rxy = np.radians(45)
        rz = np.radians(90)
        range_initial = [0.5, 0.5, 0.3, rxy, rxy, rz]
        pose_center = np.array(pose_center)
        self.initial_poses = self.generate_neighbours(initial_pose_count, range_initial, pose_center)
        self.initial_tree = None
    
    def generate_neighbours(self, count, ranges, pose):
        vs = np.empty((count, 6))
        for i in range(6):
            vs[:, i] = np.random.uniform(-ranges[i], ranges[i], size=count)
        wTc = batch_to_homogeneous_transform_with_axis_angle(pose[None, :3], pose[None, 3:])
        cTs = np.empty((count, 4, 4))
        for i in range(count):
            cTs[i] = exponential_map(vs[i], dt=1.0)
        wTs = wTc @ cTs
        return batch_to_pose_vector(wTs)
    def requires_data(self) -> bool:
        return True

    def get_image_neighbours(self, z, encoder):
        if self.initial_tree is None:
            initial_z = encoder(torch.from_numpy(self.initial_poses).float().to(z.device))
            initial_z = initial_z.cpu().numpy()
            self.initial_tree = KDTree(initial_z)
        z = z.cpu().numpy()
        nn = self.initial_tree.query(z, k=1, return_distance=False)[:, 0]
        closest_pose = self.initial_poses[nn]
        desired_neighbours = np.empty((len(closest_pose), self.random_sample_count, 6))
        for i in range(len(closest_pose)):
            desired_neighbours[i] = self.generate_neighbours(self.random_sample_count, [0.1, 0.1, 0.1, np.radians(20), np.radians(20), np.radians(20)], closest_pose[i])
        return desired_neighbours
    def on_new_desired(self, zd, encoder):
        self.desired_neighbours = self.get_image_neighbours(zd, encoder)

    def on_start_vs(self, z, encoder):
        self.starting_neighbours = self.get_image_neighbours(z, encoder)
        
    def on_end_vs(self, z):
        self.full_tree = None
        raise NotImplementedError
    @staticmethod
    def make_from_yaml(node: yaml.Node) -> PoseSampler:
        return DataBasedSampler(node['pose_center'], 1000, 1000)

def make_pose_sampler_from_yaml(node: yaml.Node) -> PoseSampler:
    res_dict = {
        'grid': GridPoseSampler.make_from_yaml,
        'grid_oversampling': GridPoseSamplerWithOversamplingCenter.make_from_yaml
    }
    return res_dict[node['type']](node)
        

class MLVS(VSMethod):

    '''
    The MLVS method, described in the paper submitted to ICRA23.
    The method is based on two networks:
        * One that processes poses (from which we obtain the interaction matrices)
        * One that processes images (from which we obtain the representations, i.e. s and s* in classical VS)

    '''
    def __init__(self, model_path: Path, name: str,
                vs_arguments: VSArguments, border: UnsignedInt = 10, dofs=6,
                gradient_method_fn: Callable[[torch.Tensor, torch.Tensor], InteractionMatrixFinder] = None,
                pose_sampler: PoseSampler = None,
                optimizer: Optional[Optimizer] = None,
                interaction_matrix_mixer: InteractionMatrixMixer = DesiredInteractionMatrix(),
                device: str = 'cuda'):
        """

        Args:
            model_path (Path): The folder which contains the .pth models
            name (str): the name of the method (for saving purposes)
            vs_arguments (VSArguments): _The VS arguments, unused for now, but kept to match api of other methods
            border (UnsignedInt, optional): The number of pixels to remove on the edges of the images. Not really relevant for this method, but kept to ensure that we have the same data as AEVS. Defaults to 10.
            dofs (int, optional): Number of controlled degrees of freedom of the camera. Should be 4 or 6. Defaults to 6.
            gradient_method_fn (Callable[[torch.Tensor, torch.Tensor], InteractionMatrixFinder], optional): Function that creates an InteractionMatrixFinder, from latent representations and their interaction matrices. Defaults to None.
            pose_sampler (PoseSampler, optional): A pose sampler, queried to generate the pose from which to find the neighbours. Defaults to None.
            optimizer (Optional[Optimizer], optional): Optimizer that computes the velocity from the error and interaction matrix. Defaults to None.
            interaction_matrix_mixer (InteractionMatrixMixer, optional). Defaults to DesiredInteractionMatrix().
            device (str, optional): 'cpu' or 'cuda'. Defaults to 'cuda'.
        """        
        self.device = device
        model_name = model_path.name
        self.image_encoder = torch.load(str(model_path / f'{model_name}_image.pth'), map_location=torch.device(self.device))
        self.pose_encoder = torch.load(str(model_path / f'{model_name}_pose.pth'), map_location=torch.device(self.device))
        self.image_encoder.eval()
        self.pose_encoder.eval()
        self.vs_arguments = vs_arguments
        self._name = name
        self.border = border

        self.dofs = dofs

        self.pose_sampler = pose_sampler
        self.poses = pose_sampler().float().to(device)

        Lposes = compute_im(self.poses.cpu()).to(device)

        with torch.no_grad():
            self.zps, self.Lzps = self.pose_encoder.forward_with_interaction_matrix(self.poses, Lposes)

        self.gradient_finder = gradient_method_fn(self.zps, self.Lzps)

        self.latent_dim = self.zps.size(-1)
        del self.zps, self.Lzps

        self.optimizer = optimizer
        self.interaction_matrix_mixer = interaction_matrix_mixer
        self.interaction_matrix = None
        self.Id_z, self.I_z = None, None
        # self.dbs = DataBasedSampler([0, 0, -0.6, 0, 0, 0], 1000, 1000)

    
    def idw_im(self, zI):
        # print(zI)
        B, K = zI.size(0), self.Lzps.size(0)
        dists = torch.sum((self.zps[None] - zI.unsqueeze(1)) ** 2, dim=-1) # B x K
        min_dist = torch.min(dists, -1)[0]
        threshold = torch.max(torch.tensor(0.2, device=zI.device), min_dist).unsqueeze(1)
        threshold_mask = (dists <= threshold)
        
        dists = dists * threshold_mask
        dists_inv = 1 / (dists + 1e-7) # In IDW, since we use mse, equivalent to p=2
        dists_inv_sum = torch.sum(dists_inv, dim=-1)

        Lidw = torch.sum(dists_inv.view(B, K, 1, 1) * self.Lzps.unsqueeze(0),dim=1) / dists_inv_sum.view(B, 1, 1)
        return Lidw
    

    def on_end_iter(self, iter_idx) -> None:
        self.optimizer.on_iter(iter_idx)
    def name(self) -> str:
        return self._name
    def error_shape(self) -> UnsignedInt:
        return self.latent_dim
    def on_new_batch(self, batch_idx, Id_processed, _desired_pose=None) -> None:
        self.interaction_matrix_mixer.reset()
        self.optimizer.reset()
        with torch.no_grad():
            self.Id_z = self.encode_images(Id_processed)
            
            # self.dbs.on_new_desired(self.Id_z, self.pose_encoder)
            if self.interaction_matrix_mixer.requires_Lid():
                self.interaction_matrix = self.gradient_finder(self.Id_z)
                # self.interaction_matrix = self.idw_im(self.Id_z)
            else:
                self.interaction_matrix = None
            ops = []
            
            if self.interaction_matrix is not None:
                ops = [('save_array', (['interaction_matrix.txt'] * self.interaction_matrix.size()[0], self.interaction_matrix.cpu().numpy()), False)]

            return ops

    def encode_images(self, I):
        b = self.border
        I = I[:, :, b:-b, b:-b].contiguous()
        return self.image_encoder(I)
    def process_image(self, images) -> ImageTorchArray:
        with torch.no_grad():
            img = to_gray(images)
            img = torch.tensor(img, requires_grad=False, device=self.device)
            img = self.image_encoder.preprocess(img).unsqueeze(1).float()
            return img
    def compute_vc(self, I_processed, _Id_processed, _iter, run_indices=None, current_pose=None):
        with torch.no_grad():
            I_z = self.encode_images(I_processed)
            L_z = None

            if self.interaction_matrix_mixer.requires_Li():
                L_z = self.gradient_finder(I_z)
                # L_z = self.idw_im(I_z)
            I_z = I_z.view(I_z.size()[0], -1)
            L_zd = self.interaction_matrix
            Id_z = self.Id_z
            if run_indices is not None:
                L_zd = self.interaction_matrix[run_indices]
                Id_z = self.Id_z[run_indices]
            L = self.interaction_matrix_mixer.compute_final_L(L_z, L_zd)

            error = (I_z - Id_z).view(Id_z.size()[0], -1)
            if self.dofs == 4:
                vc = self.optimizer(L[:, :, [0, 1, 2, 5]], error)
                vc = torch.cat((vc[:, :3], torch.zeros((vc.size(0), 2), device=vc.device), vc[:, 3:]), dim=-1)
            elif self.dofs == 6:
                vc = self.optimizer(L, error)
            
            return vc, error.cpu().numpy(), []
    
    @staticmethod
    def make_method_from_yaml(node: yaml.Node, defaults: yaml.Node, globals: yaml.Node, device: str) -> Callable[[VSArguments], 'VSMethod']:
        root_nnimvs_path = Path(globals['mlvs_models_folder'])

        return lambda args: MLVS(
            root_nnimvs_path / node['model_path'],
            node['name'],
            args,
            io.get_or_default('border', node, defaults),
            node['dofs'],
            make_interaction_matrix_finder_from_yaml(node['gradient_method']),
            make_pose_sampler_from_yaml(node['pose_sampler']),
            io.optimizer_from_yaml(io.get_or_default('optimizer', node, defaults), device),
            io.interaction_matrix_mixer_from_yaml(io.get_or_default('interaction_matrix_mixer', node, defaults)),
            device
        )

class CNNAbsolutePBVS(VSMethod):
    '''
    Method that implements VS with a neural network that performs absolute pose regression.
    The network evaluates the pose for the desired image (once) and the current image (at each iteration). 
    The pose difference is then computed and fed to a PBVS control law
    '''
    def __init__(self, model_path: Path, name: str,
                vs_arguments: VSArguments, border: UnsignedInt = 10,
                device: str = 'cuda'):
        self.device = device
        model_name = model_path.name
        self.image_encoder = torch.load(str(model_path / f'{model_name}_image.pth'), map_location=torch.device(self.device))
        self.image_encoder.eval()
        self.vs_arguments = vs_arguments
        self._name = name
        self.border = border
        self.cdTo = None
        

    def on_end_iter(self, iter_idx) -> None:
        pass
    def name(self) -> str:
        return self._name
    def error_shape(self) -> UnsignedInt:
        return 6
    def on_new_batch(self, batch_idx, Id_processed, _desired_pose=None) -> None:
        with torch.no_grad():
            orcd = self.encode_images(Id_processed)
            np_orcd = orcd.cpu().numpy()
            oTcd = batch_to_homogeneous_transform_with_axis_angle(np_orcd[:, :3], np_orcd[:, 3:])
            self.cdTo = batch_homogeneous_inverse(oTcd)
            ops = []
            return ops

    def encode_images(self, I):
        b = self.border
        I = I[:, :, b:-b, b:-b].contiguous()
        return self.image_encoder(I)
    def process_image(self, images) -> ImageTorchArray:
        with torch.no_grad():
            img = to_gray(images)
            img = torch.tensor(img, requires_grad=False, device=self.device)
            img = self.image_encoder.preprocess(img).unsqueeze(1).float()
            return img
    def compute_vc(self, I_processed, _Id_processed, _iter, run_indices=None, _current_pose=None):
        with torch.no_grad():
            orc = self.encode_images(I_processed)
            oTc = orc.cpu().numpy()
            oTc = batch_to_homogeneous_transform_with_axis_angle(oTc[:, :3], oTc[:, 3:])
            cdTc = np.matmul(self.cdTo, oTc)

            cdrc = batch_to_pose_vector(cdTc)
            
            vc = np.empty((cdTc.shape[0], 6))
            vc[:, :3] = (cdTc[:, :3, :3].transpose(0, 2, 1) @ cdrc[:, :3, None])[..., 0]
            vc[:, 3:] = cdrc[:, 3:]
            return torch.from_numpy(-vc), cdrc, []
    
    @staticmethod
    def make_method_from_yaml(node: yaml.Node, defaults: yaml.Node, globals: yaml.Node, device: str) -> Callable[[VSArguments], 'VSMethod']:
        root_nnimvs_path = Path(globals['mlvs_models_folder'])

        return lambda args: CNNAbsolutePBVS(
            root_nnimvs_path / node['model_path'],
            node['name'],
            args,
            io.get_or_default('border', node, defaults),
            device
        )

io.model_builders['MLVS'] = MLVS.make_method_from_yaml
io.model_builders['PBVSCNN'] = CNNAbsolutePBVS.make_method_from_yaml

