import cv2
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    H, W = 320, 240

    grad = np.zeros((H, W, 2))

    grad[..., 0] = (1 / 0.6) * 0.1

    px = 500
    grad[..., 0] *= px

    
    print(grad)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(grad[..., 0], grad[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = mag
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    print(rgb)
    plt.figure()
    plt.imshow(rgb)
    plt.show()
