'''
Generate plots for MLVS: study of the latent space
'''

from random import sample
from tkinter import TRUE
from typing import List
from inference.utils import to_gray
import numpy as np
import torch
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'text.latex.preamble' : [r'\usepackage{amsmath}']})
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True

from metric_learning.model.models import ImageEncoder, PoseEmbedder
from metric_learning.inference.methods import compute_im as compute_im
from geometry import *
from generator import MultiGeneratorHandler, SceneSet
from sklearn.decomposition import PCA

sys.path.append(os.getcwd())

def make_generator():
    data_root = Path('/local/sfelton/generator_scenes/').expanduser()
    scene_list = [str(data_root / 'scene_real_lower_res.jpg')]
    seed = 8193
    base_stds = np.asarray([0.1, 0.1, 10.0, 10.0])
    generator_parameters = {
        # 'base_seed': [seed * (i + 1) for i in range(num_workers)],
        'base_seed': seed,
        'lambda': 1,  # unimportant here
        'half_length_m': 0.35,
        'half_width_m': 0.5,
        'image_paths': scene_list,
        'gaussian_sets_sigmas': [base_stds],
        'gaussian_sets_probabilities': [1.0],
        'max_translation_sides': 0.1,
        'base_camera_height': 0.6,
        'h': 234,
        'w': 234,
        'px': 300,
        'py': 300,
        'u0': 124,
        'v0': 114,
        'augmentation_on_two_images': True,
        'max_translation_height': 0.08,
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
        'scene_cutout_max_count': 3,
        'desired_pose_max_rotation_xy': 0.5
    }

    s = SceneSet(generator_parameters['image_paths'])
    generator = MultiGeneratorHandler(1, s, generator_parameters)
    return generator, s


def get_image_and_encode(g: MultiGeneratorHandler, p, encoder: ImageEncoder):
    if isinstance(p, torch.Tensor):
        p = p.numpy()
    if len(p.shape) == 1:
        p = p[None, :]
    b = 5
    res = g.image_at_poses(1, 234, 234, 0.0, p, [0], False)
    return encode_image(res[0].image(), encoder)
    
def encode_image(image, encoder):
    img = preprocess_image(image, encoder)
    zi = encoder(img)
    return zi[0]

def preprocess_image(image, encoder):
    b = 5
    img = image
    img = to_gray(img[None, ...])[0]
    img = img[None, None, b:-b, b:-b]
    # fig = plt.figure()
    # plt.imshow(img[0, 0])
    # plt.show()
    return encoder.preprocess(torch.from_numpy(img).float())

def find_pose_from_image_rep(zi, pose_encoder):
    with torch.enable_grad():
        from torch.optim import SGD
        zi = zi.view(-1, zi.size(-1))
        poses = torch.zeros(zi.size(0), 6)
        poses.requires_grad = True
        optimizer = SGD([poses], lr=0.01, momentum=0.8, nesterov=True)
        for i in range(10000):
            optimizer.zero_grad()
            zp = pose_encoder(poses)
            loss = torch.nn.functional.mse_loss(zp, zi)
            loss.backward()
            if i % 10 == 0:
                print(loss.item())
            optimizer.step()
            with torch.no_grad():
                poses[:, 2:] = 0

    print(poses)
    return poses.detach()

def grad_cam_video(traj_path, image_encoder):
    from PIL import Image
    desired_image = np.array(Image.open(traj_path / 'images/Id.jpg'))
    zd = encode_image(desired_image, image_encoder)
    print(zd)
    i = 0
    class EmbeddingDistance:
        def __init__(self, zd):
            self.zd = zd
        
        def __call__(self, model_output):
            return torch.nn.functional.mse_loss(self.zd, model_output)
    
    target = [EmbeddingDistance(zd)]
    
    print(image_encoder)
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    cam = GradCAM(model=image_encoder,
             target_layers=[image_encoder.encoder.op.layer4.op[-1].op.conv2.op],
             use_cuda=False)
    visualizations = []
    while True:
        filename = 'I{0:05d}.jpg'.format(i)
        path = traj_path / 'images' / filename
        if not path.exists():
            print('breaking')
            break
        
        image = np.array(Image.open(path))
        image_preprocessed = preprocess_image(image, image_encoder)
        b = 5
        image = image[b:-b,b:-b]

        with torch.enable_grad():
            cam_gray = cam(image_preprocessed, target)[0]
            visualizations.append(show_cam_on_image(image[:, :, None] / 255, cam_gray, use_rgb=True))
        i += 1

    def vidwrite(fn, images, framerate=60, vcodec='libx264'):
        import ffmpeg
        if not isinstance(images, np.ndarray):
            images = np.asarray(images)
        print(images.shape)
        n,height,width, channels = images.shape
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
                .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
        for frame in images:
            process.stdin.write(
                frame
                    .astype(np.uint8)
                    .tobytes()
            )
        process.stdin.close()
        process.wait()
    vidwrite(str(traj_path / 'gradcam.mkv'), visualizations, framerate=24)

def make_multi_trajectory_plot(directory: Path, sample_names: List[str], pose_encoder, save_path):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image
    fig = plt.figure(figsize=(16, 16))
    
    errors = []
    ims = []

    for s in sample_names:
        p = directory / s
        errors.append(np.load(p / 'error.txt.npz')['arr_0'])
        I = Image.open(p / 'images/orig_I0.png')
        I = np.array(I)
        I = I[5:-5, 5:-5]
        ims.append(I)
    errors = np.array(errors)
    pca = PCA(2).fit(errors.reshape(-1, errors.shape[-1]))
    print(pca.explained_variance_ratio_)
    # Some offsets to avoid plotted images overlapping interesting data
    image_offsets = [np.array([0, 0]) for _ in range(len(sample_names))]
    image_offsets[0] = [0.05, 0.1]
    image_offsets[1] = [-0.01, -0.0]
    image_offsets[2] = [0.05, 0.02]
    
    image_offsets[3] = [-0.15, -0.1]
    image_offsets[-2] = [-0.05, -0.05]
    image_offsets[-3] = [-0.05, -0.05]
    
    image_offsets[-4] = [-0.05, 0.1]
    
    image_offsets[-1] = [-0.15, 0.05]
    
    for i in range(errors.shape[0]):
        err_i_pca = pca.transform(errors[i])
        sca = plt.scatter(err_i_pca[:, 0], err_i_pca[:, 1], s=28)
        plt.draw()
        color = sca.get_facecolors()[0].tolist()
        imagebox = OffsetImage(ims[i], zoom = 0.4)
        xy = err_i_pca[0] + err_i_pca[0] / np.linalg.norm(err_i_pca[0]) * 0.1 + image_offsets[i]
        ab = AnnotationBbox(imagebox, xy, frameon = True, bboxprops=dict(edgecolor=color, linewidth=7), pad=0.0)
        fig.gca().add_artist(ab)
        ab.set_zorder(100000)
        

    npose_samples = 1000
    radii = [0.2, 0.1, 0.05, 0.01, 0.005]
    center_pose = np.array([0.0, 0.0, -0.6, 0.0, 0.0, 0.0])
    with torch.no_grad():
        center_rep = pose_encoder(torch.from_numpy(center_pose[None]).float()).cpu().numpy()
        for radius in radii:
            thetas = np.linspace(0, np.radians(360), num=npose_samples)
            xs = radius * np.cos(thetas)
            ys = radius * np.sin(thetas)
            zero = np.zeros_like(xs)
            poses = np.array([xs, ys, zero, zero, zero, zero]).transpose((1, 0)) + center_pose[None]
            encoded_poses = pose_encoder(torch.from_numpy(poses).float()).cpu().numpy()
            pose_pcas = pca.transform(encoded_poses - center_rep)
            plt.plot(pose_pcas[:, 0], pose_pcas[:, 1], linestyle='--', c='k')
            if radius >= 0.01:
                if radius == radii[0]:
                    text = fr'$\lVert^{{c^*}}\mathbf{{t}}_{{c}}\rVert_2 = \text{{{int(radius * 100)}}}cm$'
                else:
                    text = fr'{int(radius * 100)}cm'
                
                chosen_point = pose_pcas[int(npose_samples * 0.2)]
                xy = chosen_point + chosen_point / np.linalg.norm(chosen_point) * 0.02
                a = plt.annotate(text, xy, fontsize=28)
                a.set_zorder(1000)
        

    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path / 'latent_trajectories.pdf')


if __name__ == '__main__':
    import faulthandler
    faulthandler.enable()
    models_path = Path('/local/sfelton/models/mlvs')
    model_name = 'embedding_model_6_dof_resnet34_180_invariance_loss'
    save_path = Path('/local/sfelton/embedding') / model_name
    save_path.mkdir(exist_ok=True)
    model_path = models_path / model_name

    pose_encoder_path = model_path / f'{model_name}_pose.pth'
    image_encoder_path = model_path / f'{model_name}_image.pth'
    

    
    g, _ = make_generator()
    print('Loading model...')
    pose_encoder = torch.load(pose_encoder_path, map_location='cpu')
    image_encoder = torch.load(image_encoder_path, map_location='cpu')
    pose_encoder.eval()
    image_encoder.eval()
    t_range = 0.2
    rxy_range = np.radians(45)
    rz_range = np.radians(90)
    dim = 25 # Use dimensions with an odd value, otherwise the desired pose will not be at the center of the plots

    range_per_dof = [t_range, t_range, t_range, rxy_range, rxy_range, rz_range]
    dof_couples = [(0, 1), (1, 3), (2, 5), (0, 2)]
    plot_per_component = False
    
    # traj_dir = Path('/local/sfelton/nnimvs/vs_results/MLVS_k=50_oversampling__chosen')
    # make_multi_trajectory_plot(traj_dir, [str(i) for i in range(8)], pose_encoder, save_path)
    for i in range(1):
        print('grad_cam')
        p = Path('/local/sfelton/mlvs_exp/p7_best_1200')
        grad_cam_video(p / str(i), image_encoder)
    

    for dofs in dof_couples:
        dofs_save_path = save_path / f'{dofs[0]}_{dofs[1]}'
        dofs_save_path.mkdir(exist_ok=True)
        dr = [range_per_dof[dofs[0]], range_per_dof[dofs[1]]]
        print(f'Generating plots for dofs: {dofs}')
        pd = torch.zeros((1, 6)).float()
        pd[:, 2] = -0.6
        # pd[:, 5] = np.radians(180)
        #Compute latent rep and interaction matrix for desired pose (at the center of the grid)
        L = compute_im(pd)
        zid = get_image_and_encode(g, pd, image_encoder)
        zd, Lzd = pose_encoder.forward_with_interaction_matrix(pd, L)
        Z = zd.size(-1)
        print(zd.size())
        
        print('Generating Data...')
        print(f'Samples per dim: {dim}, range per dim: [{-dr[0]}, {dr[0]}], [{-dr[1]}, {dr[1]}]')
        plot_extent = [-dr[0], dr[0], -dr[1], dr[1]]
        X = np.linspace(-dr[0], dr[0], num=dim)
        Y = np.linspace(-dr[1], dr[1], num=dim)
        X, Y = np.meshgrid(X, Y, indexing='ij')
        zs = torch.empty((dim, dim, zd.size(-1)))
        zsi = torch.empty((dim, dim, zd.size(-1)))
        Lzs = torch.empty((dim, dim, zd.size(-1), 6))
        
        # Compute for all points on grid
        with torch.no_grad():
            for i in range(len(X)):
                for j in range(len(Y)):
                    r = torch.zeros((1, 6)).float()
                    r[:, 2] = -0.6
                    r[:, dofs[0]] += X[i, j]
                    r[:, dofs[1]] += Y[i, j]
                    zsi[i, j] = get_image_and_encode(g, r, image_encoder)
                    LL = compute_im(r)
                    zp, LL = pose_encoder.forward_with_interaction_matrix(r, LL)
                    zs[i, j], Lzs[i, j] = zp[0], LL[0]

            error_zp_zpd = torch.sqrt(torch.sum((zs - zd.unsqueeze(0)) ** 2, dim=-1))
            error_zi_zpd = torch.sqrt(torch.sum((zsi - zd.unsqueeze(0)) ** 2, dim=-1))
            error_zi_zid = torch.sqrt(torch.sum((zsi - zid.unsqueeze(0)) ** 2, dim=-1))
            vpp = torch.empty((dim, dim, 2))
            vii = torch.empty((dim, dim, 2))
            Lzsinv = torch.pinverse(Lzs)
            for i in range(dim):
                for j in range(dim):
                    vpp[i, j] = -torch.matmul(Lzsinv[i, j], (zs[i, j] - zd[0]).unsqueeze(-1))[[dofs[0], dofs[1]], 0]
                    vii[i, j] = -torch.matmul(Lzsinv[i, j], (zsi[i, j] - zid).unsqueeze(-1))[[dofs[0], dofs[1]], 0]
            print('Generating error functions...')
            def plot_error(error, title, fig_name, v=None):
                fig = plt.figure()
                plt.imshow(np.transpose(error, (1, 0)), cmap='plasma', extent=plot_extent, aspect=dr[0] / dr[1])
                plt.colorbar()
                if v is not None:
                    plt.quiver(X, Y, v[:, :, 0], v[:, :, 1], angles='xy', units='xy')
                plt.title(title)
                plt.savefig(dofs_save_path / fig_name)
                plt.close()
            plot_error(error_zp_zpd, r'Pose-pose error: $\lVert\mathbf{z^r} - \mathbf{z^{r^*}}\rVert$', 'pose_pose_error.pdf', vpp)
            plot_error(error_zi_zpd, r'image-pose error: $\lVert\mathbf{z^I} - \mathbf{z^{r^*}}\rVert$', 'image_pose_error.pdf')
            plot_error(error_zi_zid, r'Image-image error: $\lVert\mathbf{z^I} - \mathbf{z^{I^*}}\rVert$', 'image_image_error.pdf', vii)


            def cumsum_explained_variance_plot(pca):
                cumsum = pca.explained_variance_ratio_.cumsum()
                fig = plt.figure()
                plt.title('PCA: Cumulative explained variance')
                plt.plot(cumsum)
                plt.grid()
                plt.show()
                plt.close()
            #PCA Visualization
            
            def pca_plot(z1, title, fig_name, z1_label, z1c, z2=None, z2_label=None, z2c=None):
                z1 = z1.reshape((-1, Z))
                if z2 is not None:
                    z2 = z2.reshape((-1, Z))
                tt_z = z1 if z2 is None else np.concatenate((z1, z2), axis=0)
                pca = PCA(Z, svd_solver='full').fit(tt_z)
                fig, ax = plt.subplots(1, 2)
                z1_pca = pca.transform(z1)[:, :2]
                ax[0].scatter(z1_pca[:, 0], z1_pca[:, 1], c=z1c, s=2, label=z1_label)
                # ax[0].set_box_aspect(1)
                # ax[1].set_box_aspect(1)
                
                if z2 is not None:
                    z2_pca = pca.transform(z2)[:, :2]
                    ax[0].scatter(z2_pca[:, 0], z2_pca[:, 1], c=z2c, s=2, label=z2_label)
                    ax[0].plot([z1_pca[:, 0], z2_pca[:, 0]], [z1_pca[:, 1], z2_pca[:, 1]])
                ax[0].legend()
                ax[0].set_title(title)
                cumsum = pca.explained_variance_ratio_.cumsum()
                ax[1].set_title('Cumulative explained variance ratio')
                plt.plot(cumsum)
                plt.grid()
                plt.savefig(dofs_save_path / fig_name)


            print('Generating PCA visualizations...')
            pca_plot(zs, '2-PCA on images and poses', 'pca_full.pdf', r'$\mathbf{z^r}$', 'r', zsi, r'$\mathbf{z^I}$', 'b')
            pca_plot(zs, '2-PCA on poses only', 'pca_poses.pdf', r'$\mathbf{z^r}$', 'r')
            pca_plot(zsi, '2-PCA on images only', 'pca_images.pdf', r'$\mathbf{z^I}$', 'b')
            

            if plot_per_component:
                comp_path = dofs_save_path / 'errors_component'
                comp_pose_path = comp_path / 'pose_pose'
                comp_image_path = comp_path / 'image_pose'
                comp_image_image_path = comp_path / 'image_image'
                
                for d in [comp_path, comp_pose_path, comp_image_path, comp_image_image_path]:
                    d.mkdir(exist_ok=True)
                
                print('Generating error plots per component...')
                for i in range(zs.size(-1)):
                    fig = plt.figure()
                    error_zp_zpd = torch.sqrt((zs[:, :, i] - zd[0, i]) **2)
                    error_zi_zpd = torch.sqrt((zsi[:, :, i] - zd[0, i]) ** 2)
                    error_zi_zid = torch.sqrt((zsi[:, :, i] - zid[i]) ** 2)
                    
                    plt.imshow(error_zp_zpd, cmap='plasma', extent=plot_extent)
                    plt.colorbar()
                    plt.quiver(X, Y, Lzs[:, :, i, 0], Lzs[:, :, i, 1], angles='xy')
                    plt.savefig(comp_pose_path / f'err_{i}.pdf')
                    plt.close()
                    plt.imshow(error_zi_zpd, cmap='plasma', extent=plot_extent)
                    plt.colorbar()
                    plt.quiver(X, Y, Lzs[:, :, i, 0], Lzs[:, :, i, 1], angles='xy')
                    plt.savefig(comp_image_path / f'err_{i}.pdf')
                    plt.close()
                    plt.imshow(error_zi_zid, cmap='plasma', extent=plot_extent)
                    plt.colorbar()
                    plt.quiver(X, Y, Lzs[:, :, i, 0], Lzs[:, :, i, 1], angles='xy')
                    plt.savefig(comp_image_image_path / f'err_{i}.pdf')
                    plt.close()

