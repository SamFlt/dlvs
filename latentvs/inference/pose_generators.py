from collections import namedtuple
from typing import List
import numpy as np

from geometry import *
from utils.datasets import compute_rotation_matrix_look_at


class PoseGeneratorResults():
    def __init__(self, starting_poses, starting_images, desired_poses, desired_images, scene_indices):
        self.starting_poses = starting_poses
        self.starting_images = starting_images
        self.desired_poses = desired_poses
        self.desired_images = desired_images
        self.scene_indices = scene_indices
        self.overlap = None
        self.looked_at_points = None
        self.looked_at_desired = None

    @staticmethod
    def concatenate(results: List['PoseGeneratorResults']) -> 'PoseGeneratorResults':
        res = PoseGeneratorResults(
            np.concatenate([r.starting_poses for r in results], axis=0),
            np.concatenate([r.starting_images for r in results], axis=0),
            np.concatenate([r.desired_poses for r in results], axis=0),
            np.concatenate([r.desired_images for r in results], axis=0),
            np.concatenate([r.scene_indices for r in results], axis=0),
        )
        res.overlap = np.concatenate([r.overlap for r in results], axis=0)
        if results[0].looked_at_points is not None:
            res.looked_at_points = np.concatenate([r.looked_at_points for r in results], axis=0)
            res.looked_at_desired = np.concatenate([r.looked_at_desired for r in results], axis=0)
        return res


class PoseGenerator():
    def __init__(self, generator, num_samples, batch_size, h, w, augmentation_factor):
        self.generator = generator
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.h = h
        self.w = w
        self.augmentation_factor = augmentation_factor
    def __call__(self) -> PoseGeneratorResults:
        raise NotImplementedError

class SpecificPoseGenerator(PoseGenerator):
    def __init__(self, base_args, starting_poses, desired_poses, ):
        super().__init__(*base_args)
        self.starting_poses = starting_poses
        self.desired_poses = desired_poses
    def _generate(self, poses):
        data = self.generator.image_at_poses(
            len(poses),
            self.h,
            self.w,
            self.augmentation_factor,
            self.starting_poses,
            [0 for _ in range(len(poses))],
            False,
        )
        p = np.array([s.pose_vector() for s in data])
        Is = np.array([s.image() for s in data])
        return p, Is
    def __call__(self) -> PoseGeneratorResults:

        sp, si = self._generate(self.starting_poses)
        dp, di = self._generate(self.desired_poses)
        overlap = np.zeros((len(sp), 1))

        res = PoseGeneratorResults(sp, si, dp, di, [0 for _ in range(len(sp))])
        res.overlap = overlap
        return res

class RandomPoseGenerator(PoseGenerator):
    def __init__(self, base_args):
        super().__init__(*base_args)
    
    def __call__(self) -> PoseGeneratorResults:
        ns, bs, h, w = self.num_samples, self.batch_size, self.h, self.w
        sp = np.empty((ns, 6))
        dp = np.empty((ns, 6))
        si = np.empty((ns, h, w, 3), dtype=np.uint8)
        di = np.empty((ns, h, w, 3), dtype=np.uint8)
        initial_overlaps = np.empty((ns, 1), dtype=np.float)

        for i in range(ns // bs):
            data = self.generator.new_vs_examples(
                bs, h, w, self.augmentation_factor, False, False
            )
            for j in range(bs):
                sp[i * bs + j] = data[j].trajectory_pose_vectors()[0]
                dp[i * bs + j] = data[j].desired_pose_vector()
                si[i * bs + j] = data[j].trajectory_images()[0]
                di[i * bs + j] = data[j].desired_image()
                initial_overlaps[i * bs + j] = data[j].initial_overlap()

        res = PoseGeneratorResults(sp, si, dp, di, [0 for _ in range(len(sp))])
        res.overlap = initial_overlaps
        return res

class PoseLookingAtSamePointGenerator(PoseGenerator):
    def __init__(self, base_args, desired_pose,
                reference_point,
                uniform_rand_ranges):
        super().__init__(*base_args)
        self.desired_pose = desired_pose
        self.reference_point = reference_point
        self.uniform_rand_ranges = uniform_rand_ranges
        self.random_vec = np.array([0.0, -1.0, 0.0])
        self.random_vec = self.random_vec / np.linalg.norm(self.random_vec)

    def generate_random_points(self):
        rand_positions = np.empty((self.num_samples, 3))
        for i in range(3):
            rand_positions[:, i] = np.random.uniform(
                low=-self.uniform_rand_ranges[i],
                high=self.uniform_rand_ranges[i],
                size=(self.num_samples,),
            )
        rand_positions[:] += self.desired_pose[:3]
        return rand_positions

    def generate(self, rand_positions, Rs, scene_index=0):
        ns, bs, h, w = self.num_samples, self.batch_size, self.h, self.w
        dp = np.repeat(np.expand_dims(self.desired_pose, 0), ns, axis=0)
        orientations = batch_rotation_matrix_to_axis_angle(Rs)
        sp = np.concatenate((rand_positions, orientations), axis=-1)

        si = np.empty((self.num_samples, h, w, 3), dtype=np.uint8)

        for i in range(ns // bs):
            data = self.generator.image_at_poses(
                bs,
                h,
                w,
                self.augmentation_factor,
                sp[i * bs : (i + 1) * bs], [scene_index for _ in range(bs)],
                False
            )
            for j, sample in enumerate(data):
                si[i * bs + j] = sample.image()
        desired_image = self.generator.image_at_poses(
            1, h, w, self.augmentation_factor, dp[0:1], [scene_index], False
        )[0].image()
        di = np.repeat(np.expand_dims(desired_image, 0), ns, axis=0)
        res = PoseGeneratorResults(sp, si, dp, di, [scene_index for _ in range(ns)])
        res.overlap = np.zeros((ns, 1))
        return res
    def fill_rotation_matrix(self, position, point, Rs, index):
        forward = point - position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.random_vec)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        Rs[index, :, 0] = right
        Rs[index, :, 1] = up
        Rs[index, :, 2] = forward
    def __call__(self) -> PoseGeneratorResults:
        ns, bs, h, w = self.num_samples, self.batch_size, self.h, self.w
        rand_positions = self.generate_random_points()
        Rs = np.empty((ns, 3, 3))
        random_vec = np.array([0.0, -1.0, 0.0])
        random_vec = random_vec / np.linalg.norm(random_vec)
        for i, position in enumerate(rand_positions):
            self.fill_rotation_matrix(position, self.reference_point, Rs, i)

        return self.generate(rand_positions, Rs)

    @staticmethod
    def get_default(base_args) -> PoseGenerator:
        return  PoseLookingAtSamePointGenerator(base_args, [0.0, 0.0, -0.6, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.6, 0.6, 0.15])
        

class PoseLookingAtSamePointWithNoiseGenerator(PoseLookingAtSamePointGenerator):
    def __init__(self, base_args, desired_pose, reference_point, uniform_rand_ranges, mode, 
                look_at_rand_range = None, num_circles = None, circle_radius_aug = None, scene_index=0):
        super().__init__(base_args, desired_pose, reference_point,
                uniform_rand_ranges)
        self.scene_index = scene_index
        assert mode in ['circle', 'random']
        self.mode = mode
        if mode == 'circle':
            self.num_circles = num_circles
            assert (
                self.num_samples % self.num_circles == 0
            ), "Number of samples must be dividable by number of circles"
            self.circle_radius_aug = circle_radius_aug
            self.samples_per_circle = self.num_samples // self.num_circles
        elif mode == 'random':
            self.look_at_rand_range = look_at_rand_range

    def sample_points_circle(self, radius):
        rand_theta = np.random.uniform(-np.pi, np.pi, size=self.samples_per_circle)
        x = np.cos(rand_theta) * radius + self.reference_point[0]
        y = np.sin(rand_theta) * radius + self.reference_point[1]
        z = np.repeat(self.reference_point[2], self.samples_per_circle)
        return np.column_stack((x, y, z))
    def sample_points_uniform(self, rand_range, ref_point):
        x = np.random.uniform(
        low=-rand_range,
        high=rand_range,
        size=(self.num_samples, 1),
        )
        y = np.random.uniform(
            low=-rand_range,
            high=rand_range,
            size=(self.num_samples, 1),
        )
        z = np.zeros((self.num_samples, 1))
        points = np.concatenate((x, y, z), axis=-1)
        return points + ref_point
    
    def __call__(self) -> PoseGeneratorResults:
        ns = self.num_samples

        rand_positions = self.generate_random_points()
        looked_at_points = np.empty((ns, 3))
        looked_at_desired = np.repeat(
            np.expand_dims(np.array(self.reference_point), 0), ns, axis=0
        )
        Rs = np.empty((ns, 3, 3))
        random_vec = np.array([0.0, -1.0, 0.0])
        random_vec = random_vec / np.linalg.norm(random_vec)
        
        if self.mode == 'circle':

            assert (
                self.num_samples % self.num_circles == 0
            ), "Number of samples must be dividable by number of circles"
            samples_per_circle = ns // self.num_circles
            for cn in range(self.num_circles):
                radius = self.circle_radius_aug * (cn + 1)
                istart = cn * samples_per_circle
                points = self.sample_points_circle(radius)
                looked_at_points[istart : istart + samples_per_circle] = points
        else:
            looked_at_points = self.sample_points_uniform(self.look_at_rand_range, self.reference_point)


        for i, position in enumerate(rand_positions):
            p = looked_at_points[i]
            self.fill_rotation_matrix(position, p, Rs, istart + i)

        res = self.generate(rand_positions, Rs, self.scene_index)
        res.looked_at_desired = looked_at_desired
        res.looked_at_points = looked_at_points
        return res
    @staticmethod
    def get_default(base_args, mode) -> PoseGenerator:
        return  PoseLookingAtSamePointWithNoiseGenerator(base_args, [0.0, 0.0, -0.6, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.6, 0.6, 0.15],
                                                         mode, [0.2, 0.2], 5, 0.02, scene_index=0)

class PoseLookingAtSamePointWithNoiseAndRotationZGenerator(PoseLookingAtSamePointWithNoiseGenerator):
    def __init__(self, base_args, desired_pose, reference_point, uniform_rand_ranges, num_circles, circle_radius_aug, rz_max_deg, scene_index=0):
        super().__init__(base_args, desired_pose, reference_point, uniform_rand_ranges, 'circle', None, num_circles, circle_radius_aug, scene_index)
        self.rz_max = np.radians(rz_max_deg)
    def __call__(self) -> PoseGeneratorResults:
        ns, pref = self.num_samples, self.reference_point
        rand_positions = self.generate_random_points()
        
        looked_at_points = np.empty((ns, 3))
        looked_at_desired = np.repeat(
            np.expand_dims(np.array(pref), 0), ns, axis=0
        )
        Rs = np.empty((ns, 3, 3))
        
        samples_per_circle = ns // self.num_circles

        def Rz_rotation(theta):
            # theta = np.random.uniform(-self.rz_max, self.rz_max, size=len(self.la_Rs))
            c = np.cos(theta)
            s = np.sin(theta)
            Rz = np.zeros((len(theta), 3, 3))
            Rz[:, 2, 2] = 1
            Rz[:, 0, 0] = c
            Rz[:, 1, 1] = c
            Rz[:, 0, 1] = -s
            Rz[:, 1, 0] = s
            return Rz

        for cn in range(self.num_circles):
            radius = self.circle_radius_aug * (cn + 1)
            istart = cn * samples_per_circle
            points = self.sample_points_circle(radius)
            looked_at_points[istart : istart + samples_per_circle] = points
            la_Rs = np.transpose(compute_rotation_matrix_look_at(points, rand_positions[istart : istart + samples_per_circle]), (0, 2 ,1))
            theta = np.linspace(-self.rz_max, self.rz_max, num=samples_per_circle)
            Rz = np.transpose(Rz_rotation(theta), (0, 2, 1))
            Rs[istart: istart + samples_per_circle] = np.matmul(la_Rs, Rz)
           
        res = self.generate(rand_positions, Rs, self.scene_index)
        res.overlap = np.zeros((ns, 1))
        res.looked_at_desired = looked_at_desired
        res.looked_at_points = looked_at_points
        return res
    @staticmethod
    def get_default(base_args) -> PoseGenerator:
        return  PoseLookingAtSamePointWithNoiseAndRotationZGenerator(base_args, [0.0, 0.0, -0.6, 0.0, 0.0, 0.0],
                                                                    [0.0, 0.0, 0.0], [0.6, 0.6, 0.15],
                                                                    4, 0.08, rz_max_deg=120, scene_index=0)

class PoseScrewMotionGenerator(PoseGenerator):
    def __init__(self, base_args, count_diff_desired_poses,
                tz_range, tz_step_count, rz_range, rz_step_count, scene_index=0, desired_max_rotation=180):
        super().__init__(*base_args)
        self.count_diff_desired_poses = count_diff_desired_poses
        self.scene_index = scene_index
        self.tz_range = tz_range
        self.tz_step_count =tz_step_count
        self.rz_range = rz_range
        self.rz_step_count = rz_step_count
        self.desired_max_rotation = desired_max_rotation
        self.scene_index = scene_index

    def generate_end_poses(self, d_rot):
        xs = np.random.uniform(low=-0.2, high=0.2, size=(self.count_diff_desired_poses, 1))
        ys = np.random.uniform(low=-0.2, high=0.2, size=(self.count_diff_desired_poses, 1))
        zs = np.array([-0.6 for i in range(self.count_diff_desired_poses)]).reshape(
            (self.count_diff_desired_poses, 1)
        )
        thetas = np.random.uniform(
            low=-d_rot, high=d_rot, size=(self.count_diff_desired_poses, 1)
        )
        rxys = np.zeros((self.count_diff_desired_poses, 2))
        return np.concatenate((xs, ys, zs, rxys, thetas), axis=-1)
    def make_starting_poses(self, desired_pose):
        starting_poses = np.empty((self.tz_step_count * self.rz_step_count, 6))
        starting_poses[:, :2] = desired_pose[:2]
        starting_poses[:, 3:5] = desired_pose[3:5]

        i = 0
        rzs = np.linspace(self.rz_range[0], self.rz_range[1], self.rz_step_count)
        tzs = np.linspace(self.tz_range[0], self.tz_range[1], self.tz_step_count)
        for rz in rzs:
            rz_rad = np.radians(rz)
            for tz in tzs:
                starting_poses[i, 2] = desired_pose[2] + tz
                starting_poses[i, 5] = desired_pose[5] + rz_rad
                i += 1
        return starting_poses
    def __call__(self) -> PoseGeneratorResults:
        sample_count = self.count_diff_desired_poses * self.tz_step_count * self.rz_step_count
        d_rot = np.radians(self.desired_max_rotation)

        
        def generate_images(poses):
            batch_size, h, w, = self.batch_size, self.h, self.w
            images = np.empty((len(poses), h, w, 3), dtype=np.uint8)
            b = 1 if len(poses) % batch_size != 0 else batch_size
            for i in range(len(poses) // b):
                data = self.generator.image_at_poses(
                    b, h, w, self.augmentation_factor,
                    poses[i * b : (i + 1) * b],
                    [self.scene_index for _ in range(b)],
                    False,
                )
                for j, sample in enumerate(data):
                    images[i * b + j] = sample.image()
            return images

        dp = self.generate_end_poses(d_rot)
        di = generate_images(dp)

        starting_poses_arrays = []
        for pose in dp:
            starting_poses_arrays.append(self.make_starting_poses(pose))
        sp = np.concatenate(starting_poses_arrays, axis=0)
        si = generate_images(sp)
        dp = np.repeat(dp, self.tz_step_count * self.rz_step_count, axis=0)
        di = np.repeat(di, self.tz_step_count * self.rz_step_count, axis=0)

        res = PoseGeneratorResults(sp, si, dp, di, [self.scene_index for _ in range(sample_count)])
        res.overlap = np.zeros((sample_count, 1))
        return res

    @staticmethod
    def get_default(base_args) -> PoseGenerator:
        return  PoseScrewMotionGenerator(base_args, 10, [-0.3, 0.3], 5, [-70, 70], 10, desired_max_rotation=0)

class PoseMultisceneGenerator(PoseGenerator):
    def __init__(self, base_args, pose_generator, num_scenes):
        super().__init__(*base_args)
        self.pose_generator = pose_generator
        self.num_scenes = num_scenes
    def __call__(self) -> PoseGeneratorResults:
        results = []
        for i in range(self.num_scenes):
            self.pose_generator.scene_index = i
            results.append(self.pose_generator())

        return PoseGeneratorResults.concatenate(results)




def multiscene_poses_look_at_with_noise(
    generator,
    num_samples,
    batch_size,
    h,
    w,
    augmentation_factor,
    desired_pose,
    reference_point,
    uniform_rand_ranges,
    look_at_rand_range,
    scene_count,
):
    def gen_for_scene(si):
        return poses_looking_at_same_point_with_noise_random(
            generator,
            num_samples,
            batch_size,
            h,
            w,
            augmentation_factor,
            desired_pose,
            reference_point,
            uniform_rand_ranges,
            look_at_rand_range,
            scene_index=si,
        )

    (
        starting_poses,
        starting_images,
        desired_poses,
        desired_images,
        overlaps,
        looked_at_points,
        looked_at_desired,
    ) = [[] for _ in range(7)]

    for i in range(scene_count):
        sp, si, dp, di, o, lap, lad = gen_for_scene(i)
        starting_poses.append(sp)
        starting_images.append(si)
        desired_poses.append(dp)
        desired_images.append(di)
        overlaps.append(o)
        looked_at_points.append(lap)
        looked_at_desired.append(lad)

    scene_indices = np.repeat(range(scene_count), num_samples)
    starting_poses = np.concatenate(starting_poses, axis=0)
    starting_images = np.concatenate(starting_images, axis=0)
    desired_poses = np.concatenate(desired_poses, axis=0)
    desired_images = np.concatenate(desired_images, axis=0)
    overlaps = np.concatenate(overlaps, axis=0)
    looked_at_points = np.concatenate(looked_at_points, axis=0)
    looked_at_desired = np.concatenate(looked_at_desired, axis=0)

    return (
        starting_poses,
        starting_images,
        desired_poses,
        desired_images,
        overlaps,
        looked_at_points,
        looked_at_desired,
        scene_indices,
    )




def multiscene_poses_screw_motion(
    generator,
    num_samples,
    batch_size,
    h,
    w,
    augmentation_factor,
    desired_pose,
    tz_range,
    rz_range,
    scene_count,
):
    def gen_for_scene(si):
        return poses_screw_motion_random(
            generator,
            num_samples,
            batch_size,
            h,
            w,
            augmentation_factor,
            desired_pose,
            tz_range,
            rz_range,
            scene_index=si,
        )

    starting_poses, starting_images, desired_poses, desired_images, overlaps = [
        [] for _ in range(5)
    ]

    for i in range(scene_count):
        sp, si, dp, di, o = gen_for_scene(i)
        starting_poses.append(sp)
        starting_images.append(si)
        desired_poses.append(dp)
        desired_images.append(di)
        overlaps.append(o)
    scene_indices = np.repeat(range(scene_count), num_samples)

    starting_poses = np.concatenate(starting_poses, axis=0)
    starting_images = np.concatenate(starting_images, axis=0)
    desired_poses = np.concatenate(desired_poses, axis=0)
    desired_images = np.concatenate(desired_images, axis=0)
    overlaps = np.concatenate(overlaps, axis=0)

    return (
        starting_poses,
        starting_images,
        desired_poses,
        desired_images,
        np.zeros((num_samples, 1)),
        scene_indices,
    )


def txy_motions(
    generator,
    num_samples,
    batch_size,
    h,
    w,
    augmentation_factor,
    count_diff_desired_poses,
    Z,
    motion_range,
    desired_poses=None,
):

    assert num_samples % count_diff_desired_poses == 0
    samples_per_pose = num_samples // count_diff_desired_poses

    motion_xy = np.random.uniform(
        low=-motion_range, high=motion_range, size=(samples_per_pose, 2)
    )

    def generate_end_poses():
        xs = np.random.uniform(low=-0.2, high=0.2, size=(count_diff_desired_poses, 1))
        ys = np.random.uniform(low=-0.2, high=0.2, size=(count_diff_desired_poses, 1))
        zs = np.array([-Z for i in range(count_diff_desired_poses)]).reshape(
            (count_diff_desired_poses, 1)
        )
        thetas = np.random.uniform(
            low=-np.pi, high=np.pi, size=(count_diff_desired_poses, 1)
        )
        rxys = np.zeros((count_diff_desired_poses, 2))
        return np.concatenate((xs, ys, zs, rxys, thetas), axis=-1)

    def generate_images(poses):
        images = np.empty((len(poses), h, w, 3), dtype=np.uint8)
        b = 1 if len(poses) % batch_size != 0 else batch_size
        for i in range(len(poses) // b):
            data = generator.image_at_poses(
                b,
                h,
                w,
                augmentation_factor,
                poses[i * b : (i + 1) * b],
                [0 for _ in range(b)],
                False,
            )
            for j, sample in enumerate(data):
                images[i * b + j] = sample.image()
        return images

    if desired_poses is not None:
        desired_poses = generate_end_poses()
    else:
        assert len(desired_poses) == count_diff_desired_poses
    desired_images = generate_images(desired_poses)

    def make_starting_poses(desired_pose):
        pose_diff = np.zeros((samples_per_pose, 6))
        pose_diff[:, :2] = motion_xy
        cdTw = to_homogeneous_transform_with_axis_angle(
            desired_pose[:3], desired_pose[3:]
        )
        cTcd = batch_to_homogeneous_transform_with_axis_angle(
            pose_diff[:, :3], pose_diff[:, 3:]
        )
        cTw = cTcd @ cdTw
        return batch_to_pose_vector(cTw)

    starting_poses_arrays = []
    for pose in desired_poses:
        starting_poses_arrays.append(make_starting_poses(pose))
    starting_poses = np.concatenate(starting_poses_arrays, axis=0)
    starting_images = generate_images(starting_poses)
    desired_poses = np.repeat(desired_poses, samples_per_pose, axis=0)
    desired_images = np.repeat(desired_images, samples_per_pose, axis=0)

    return (
        starting_poses,
        starting_images,
        desired_poses,
        desired_images,
        np.zeros((num_samples, 1)),
    )


def txyz_motions(
    generator,
    num_samples,
    batch_size,
    h,
    w,
    augmentation_factor,
    count_diff_desired_poses,
    Z,
    txy_range,
    tz_range,
    desired_poses=None,
):

    assert num_samples % count_diff_desired_poses == 0
    samples_per_pose = num_samples // count_diff_desired_poses

    motion_xy = np.random.uniform(
        low=-txy_range, high=txy_range, size=(samples_per_pose, 2)
    )
    motion_z = np.random.uniform(low=-tz_range, high=tz_range, size=(samples_per_pose))

    def generate_end_poses():
        xs = np.random.uniform(low=-0.2, high=0.2, size=(count_diff_desired_poses, 1))
        ys = np.random.uniform(low=-0.2, high=0.2, size=(count_diff_desired_poses, 1))
        zs = np.array([-Z for _ in range(count_diff_desired_poses)]).reshape(
            (count_diff_desired_poses, 1)
        )
        thetas = np.random.uniform(
            low=-np.pi, high=np.pi, size=(count_diff_desired_poses, 1)
        )
        rxys = np.zeros((count_diff_desired_poses, 2))
        return np.concatenate((xs, ys, zs, rxys, thetas), axis=-1)

    def generate_images(poses):
        images = np.empty((len(poses), h, w, 3), dtype=np.uint8)
        b = 1 if len(poses) % batch_size != 0 else batch_size
        for i in range(len(poses) // b):
            data = generator.image_at_poses(
                b,
                h,
                w,
                augmentation_factor,
                poses[i * b : (i + 1) * b],
                [0 for _ in range(b)],
                False,
            )
            for j, sample in enumerate(data):
                images[i * b + j] = sample.image()
        return images

    if desired_poses is None:
        desired_poses = generate_end_poses()
    else:
        desired_poses = np.array(desired_poses)
        assert len(desired_poses) == count_diff_desired_poses
    desired_images = generate_images(desired_poses)

    def make_starting_poses(desired_pose):
        pose_diff = np.zeros((samples_per_pose, 6))
        pose_diff[:, :2] = motion_xy
        pose_diff[:, 2] = motion_z

        wTcd = batch_to_homogeneous_transform_with_axis_angle(
            desired_pose[None, :3], desired_pose[None, 3:]
        )[0]
        cdTc = batch_to_homogeneous_transform_with_axis_angle(
            pose_diff[:, :3], pose_diff[:, 3:]
        )
        wTc = wTcd @ cdTc
        return batch_to_pose_vector(wTc)

    starting_poses_arrays = []
    for pose in desired_poses:
        starting_poses_arrays.append(make_starting_poses(pose))
    starting_poses = np.concatenate(starting_poses_arrays, axis=0)
    starting_images = generate_images(starting_poses)
    desired_poses = np.repeat(desired_poses, samples_per_pose, axis=0)
    desired_images = np.repeat(desired_images, samples_per_pose, axis=0)

    return (
        starting_poses,
        starting_images,
        desired_poses,
        desired_images,
        np.zeros((num_samples, 1)),
    )
