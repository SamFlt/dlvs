import time
import torch
import sys
import numpy as np
# from models.common import DVS as DVSpt
# from models.common import DVSInteractionMatrix, ImageGradients
# from models.im_computable import permute_im_to_vec_rep_if_required, im_is_in_image_rep, permute_im_to_vec_rep_if_required_minimal_checks

if __name__ == '__main__':
    model_path = sys.argv[1]
    device = sys.argv[2]
    model = torch.load(model_path, map_location=torch.device(device))

    time_total = 0
    count = 1000
    for i in range(count):
        image = torch.rand((1, 1, 224, 224), device=device)
        L = torch.rand((1, 224 * 224, 6), device=device)
        start = time.time()
        _, _ = model.forward_encode_with_interaction_matrix(image, L)
        end = time.time()
        time_total += (end - start)
    time_avg = time_total / count
    print('Computing z and Lz took {} ms'.format(time_avg * 1000))



