from pathlib import Path
import torch
import matplotlib.pyplot as plt
import sys
from generator import ClusteringTripletOutput, Generator, MultiGeneratorHandler, DVS, SceneSet
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


if __name__ == "__main__":
    data_root = Path('~/code/visp_inference/data').expanduser()
    scene_list = [str(data_root / 'scene_real_lower_res.jpg')]
    scenes = SceneSet(scene_list)
    seed = 8193
    base_stds = np.asarray([0.05, 0.05, 5.0, 5.0])
    generator_parameters = {
        # 'base_seed': [seed * (i + 1) for i in range(num_workers)],
        'base_seed': seed,
        'lambda': 1,  # unimportant here
        'half_length_m': 0.3,
        'half_width_m': 0.4,
        'image_paths': scene_list,
        # 'image_paths': [str((data_root / 'scenes_cvpr09_hollywood' / 'scene_real_lower_res.jpg').expanduser())],
        # 'image_paths': [str((data_root / 'scene_natural.jpg').expanduser())],
        #'gaussian_sets_sigmas': [base_stds],
        #'gaussian_sets_probabilities': [1],
        'gaussian_sets_sigmas': [base_stds],
        'desired_pose_max_rotation_xy': 0.0,
        'gaussian_sets_probabilities': [1.0],
        'max_translation_sides': 0.0,
        'base_camera_height': 0.7,
        'px': 570,
        'py': 570,
        'u0': 119,
        'v0': 109,
        'augmentation_on_two_images': True,
        'max_translation_height': 0.0,
        'gaussian_lights_gain_std': 0.2,
        'gaussian_lights_max_count': 4,
        'gaussian_lights_base_std_scene_rel': 0.5,
        'gaussian_lights_std_spread_scene_rel': 0.2,
        'global_lighting_augmentation_std': 0.2,
        'global_lighting_augmentation_bias_std': 0.1,
        'visibility_threshold': 0.5,
        'use_scene_cutout': False,
        'scene_cutout_size_min_rel': 0.01,
        'scene_cutout_size_max_rel': 0.2,
        'scene_cutout_max_dist_from_center': 0.3,
        'scene_cutout_use_pixel_level': False,
        'scene_cutout_max_count': 3
    }
    model_path = Path(sys.argv[1])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(str(model_path), map_location=torch.device(device))
    model.eval()

    def preprocess(x):
        return (x - 127.5) / 127.5
    def unprocess(x):
        return (x * 127.5 + 127.5).astype(np.uint8)
    def ssd(x1, x2):
        return ((x1 - x2) ** 2).sum()
    def compute_diff(I1, I2):
        x = I1.astype(np.int) - I2.astype(np.int)
        x = x / 2
        x += 127
        return x.astype(np.uint8)
    save_dir = Path('/local/sfelton/ae_dvs_plot').expanduser()
    save_dir.mkdir(exist_ok=True)
    count = 70
    center = (0.0, 0.0)
    dims = (0.3, 0.3)
    Z = 0.8
    half_dim = (dims[0] / 2, dims[1] / 2)
    poses = np.zeros((count, count, 6))
    for i in range(count):
        x = center[0] - half_dim[0] + (dims[0] / count) * i
        for j in range(count):
            y = center[1] - half_dim[1] + (dims[1] / count) * j
            poses[i, j, 0] = x
            poses[i, j, 1] = y
            poses[i, j, 2] = -Z
    generator = MultiGeneratorHandler(count ** 2, scenes, generator_parameters)
    data = generator.image_at_poses(count * count, 224, 224, 0, poses.reshape((count ** 2, 6)))
    desired = generator.image_at_poses(1, 224, 224, 0, [[center[0], center[1], -Z, 0.0, 0.0, 0.0]])[0]

    images = [d.image() for d in data]
    images = np.array([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images])
    desired_image = cv2.cvtColor(desired.image(), cv2.COLOR_RGB2GRAY)
    ssds = np.sum((images - desired_image.reshape((1, 224, 224))) ** 2, axis=(1, 2))
    
    im = plt.imshow(ssds.reshape((count, count)), interpolation='bicubic', cmap=plt.cm.plasma, extent=(-half_dim[0], half_dim[0], -half_dim[1], half_dim[1]))
    plt.colorbar(im)
    plt.savefig('plot_ssd_original_images.png')
    print('Encoding decoding images')
    def encode_decode(images):
        batch_size = min(100, len(images))
        res = np.empty((len(images), 224, 224))

        for i in range(len(images) // batch_size):
            print('Image encoding: Batch', i)
            batch = images[i*batch_size:(i+1)*batch_size]
            batch = torch.from_numpy(preprocess(batch).reshape((-1, 1, 224, 224))).to(device, dtype=torch.float)
            with torch.no_grad():
                data = model(batch)
                im_decoded = data[-1]
                im_decoded = unprocess(im_decoded.cpu().numpy().reshape((-1, 224, 224))).astype(np.uint8)
                res[i*batch_size:(i+1)*batch_size] = im_decoded
        return res
    desired_decoded = encode_decode(np.expand_dims(desired_image, 0))
    images_decoded = encode_decode(images)
    print('Computing ssd for decoded images')
    def label_offset(ax, axis="y"):
        fmt = None
        set_label = None
        label = None
        if axis == "y":
            fmt = ax.yaxis.get_major_formatter()
            ax.yaxis.offsetText.set_visible(False)
            set_label = ax.set_ylabel
            label = ax.get_ylabel()

        elif axis == "x":
            fmt = ax.xaxis.get_major_formatter()
            ax.xaxis.offsetText.set_visible(False)
            set_label = ax.set_xlabel
            label = ax.get_xlabel()
        elif axis == "z":
            fmt = ax.zaxis.get_major_formatter()
            ax.zaxis.offsetText.set_visible(False)
            set_label = ax.set_zlabel
            label = ax.get_zlabel()

        def update_label(event_axes):
            offset = fmt.get_offset()
            if offset == '':
                set_label("{}".format(label))
            else:
                set_label("{} ({})".format(label, offset))
            return

        ax.callbacks.connect("ylim_changed", update_label)
        ax.callbacks.connect("xlim_changed", update_label)
        ax.figure.canvas.draw()
        update_label(None)
        return
    def plot_3d(poses, ssd, filename):
        plt.clf()
        plt.cla()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(poses[:, :, 0], poses[:, :, 1], ssd.reshape((count, count)), cmap=plt.cm.plasma,
                        linewidth=-1, edgecolors='none', edgecolor='none', antialiased=True)
        surf2 = ax.plot_surface(poses[:, :, 0], poses[:, :, 1], np.zeros((count, count)),
                        linewidth=-1, edgecolor='none', edgecolors='none', antialiased=True, shade=False, facecolors=plt.cm.plasma(ssd.reshape((count, count)).astype(np.int64) / np.amax(ssd)))
                        
        fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
        ax.set_xlabel('x translation', labelpad=12)
        ax.set_ylabel('y translation', labelpad=12)
        ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))
        ax.set_zlabel('$(vec(\\mathbf{I}) - vec(\\mathbf{I^*}))^2$', labelpad=12)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # fig.autofmt_xdate()
        label_offset(ax, "z")
        fig.tight_layout()
        plt.savefig(filename)

    ssds_decoded = np.sum((images_decoded - desired_decoded.reshape((1, 224, 224))) ** 2, axis=(1, 2))
    plt.cla()
    plt.clf()
    im = plt.imshow(ssds_decoded.reshape((count, count)), interpolation='bicubic', cmap=plt.cm.plasma, extent=(-half_dim[0], half_dim[0], -half_dim[1], half_dim[1]))
    plt.colorbar(im)
    plt.savefig('plot_ssd_decoded_images.png')
    plot_3d(poses, ssds, 'plot_3d_ssd_original_images.png')
    plot_3d(poses, ssds_decoded, 'plot_3d_ssd_decoded_images.png')
    


