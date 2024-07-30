import argparse

import numpy as np

import torch
from torch import autograd
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
def get_gradient_per_component(inp, output):
        '''
        inp: A batch of images: B x C x H x W
        outputs: a batch of representations obtained from images: B x Z
        '''
        B, C, H, W = inp.size()
        Z = output.size(-1)
        dzdI = torch.empty((B, Z, C, H, W), device=output.device)
        
        for i in range(Z):
                grad_outs = torch.zeros_like(output, device=output.device)
                grad_outs[:, i] = 1.0
                gradient = autograd.grad(outputs=output, inputs=inp,
                                 grad_outputs=grad_outs,
                                 create_graph=True, retain_graph=True,
                                 only_inputs=True, allow_unused=True)[0]
                dzdI[:, i] = gradient.detach()
        return dzdI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate reconstructions for AEVS')
    parser.add_argument('--model', type=str,
                    help='Model path to use for reconstructions (.pth file)')
    parser.add_argument('--target', type=str,
                    help='''The target folder, that is the root of a servoing example. It should contain an images folder.''')
    parser.add_argument('--no_decoding',  action='store_true',
                    help='''Whether to perform decoding (image reconstruction from latent space) or not. Disable for models that are not autoencoders.''')
    
    parser.add_argument('--jacobians', action='store_true',
                    help='''Whether to generate the network jacobians''')
    parser.add_argument('--subsample', type=int, default=1,
                    help='''Time subsampling, useful for jacobian computation''')
    
    
    border = 5

    args = parser.parse_args()

    subsample = args.subsample

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.model, map_location=device)
    root_folder = Path(args.target)
    images_folder = root_folder / 'images'
    reconstructions_folder = root_folder / 'reconstructions'
    reconstructions_folder.mkdir(exist_ok=True)
    jacobians_folder = root_folder / 'jacobians'
    jacobians_folder.mkdir(exist_ok=True)
    files = images_folder.iterdir()
    def use_image(file):
        return file.name.endswith(('.jpg', '.png')) \
            and not 'diff' in file.name \
            and not 'rec' in file.name \
            and not file.name.startswith('orig_I0.') \
            and not file.name.startswith('orig_Id.')

    image_paths = sorted(list(filter(use_image, files)))
    print(image_paths)
    image_count = len(image_paths)
    jacobians = None
    for i in range(0, image_count, subsample):
        image_file = image_paths[i]
        
        
        im = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)

        im = torch.from_numpy(im).to(device)
        im = model.preprocess(im)
        im = im[border:-border, border:-border].contiguous()
        h,w = im.size()
        im = im.view(1,1,h,w)
        print(f'Processing image {i + 1}/{image_count}: {image_file.name}...', end='\r')
        if not args.no_decoding:
            with torch.no_grad():
                z, rec = model(im)
                rec = model.unprocess(rec).cpu().numpy()[0, 0]
                cv2.imwrite(str(reconstructions_folder / image_file.name), rec)
        if args.jacobians:

            im.requires_grad = True
            z = model.forward_encode(im)
            if jacobians is None:
                jacobians = np.empty((image_count // subsample, z.size(-1), h, w))
            J = get_gradient_per_component(im, z)[0, :, 0]
            fig = plt.figure()
            grid = ImageGrid(fig, 111,
                 nrows_ncols=(J.size(0) // 8, 8),  
                 axes_pad=0.1,
            )
            J = J.cpu().numpy()
            jacobians[i // subsample] = J
            for n in range(J.shape[0]):
                grid[n].imshow(J[n], cmap='plasma')
            plt.tight_layout()
            plots_folder = jacobians_folder / 'plots'
            plots_folder.mkdir(exist_ok=True)
            plt.savefig(str(plots_folder / image_file.name))
            plt.close()

    for i in range(jacobians.shape[1]): # min-max normalize along trajectory, per z component
        minj = np.min(jacobians[:, i])
        maxj = np.max(jacobians[:, i])
        diff = maxj - minj
        jacobians[:, i] = (jacobians[:, i] - minj) / diff
        jacobians[:, i] *= 255
    
    jacobians = jacobians.astype(np.uint8)
    np.savez_compressed(str(jacobians_folder / 'jacobians.npz'), jacobians=jacobians)
