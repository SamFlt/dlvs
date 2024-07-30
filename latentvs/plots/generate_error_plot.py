import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def fig2data(fig):
    """
    Taken from: https://web-backend.icare.univ-lille.fr/tutorials/convert_a_matplotlib_figure
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    # w, h = fig.canvas.get_width_height()
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    w, h = int(bbox.width*fig.dpi), int(bbox.height*fig.dpi)
    buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
 
    buf = np.roll (buf, 3, axis = 2)
    return buf

if __name__ == '__main__':
    plt.rc('text', usetex=True)
    path = sys.argv[1]
    error_dir = Path(path) / 'errors'
    count = len(list(error_dir.iterdir()))
    errors = []
    for i in range(count):
        d = error_dir / '{}.npy'.format(i)
        error = np.load(str(d))
        # print(error)
        errors.append(error)
    fig = plt.figure()
    plt.plot(errors)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlim([-10, count])
    plt.tight_layout()
    plt.savefig(str(Path(path) / 'error_plot.pdf'))
    print('Final image was saved')

    errors = np.array(errors)
    print(errors.shape)
    error_norm = np.linalg.norm(errors, axis=-1)
    print(error_norm.shape)
    fig = plt.figure()
    plt.plot(error_norm)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Error norm', fontsize=14)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlim([-10, count])
    plt.hlines(0, -10, count, colors='k')
    plt.tight_layout()
    plt.savefig(str(Path(path) / 'error_norm.pdf'))
    print('Saved norm error')


    plt.close()
    frames = []
    for i in range(count):
        fig = plt.figure()
        plt.plot(errors[:(i+1)])
        plt.xlabel('Iterations', fontsize=16)
        plt.ylabel('Error', fontsize=16)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.xlim([-10, count])
        plt.hlines(0, -10, count, colors='k')
        
        plt.tight_layout()
        frame = fig2data(fig)
        frames.append(frame)

        plt.close()
        print('Generated frame {}/{}'.format(i + 1, count), end='\r')
    frames = np.array(frames)
    print(frames.shape)
    import ffmpeg
    def vidwrite(fn, images, framerate=60, vcodec='libx264'):
        if not isinstance(images, np.ndarray):
            images = np.asarray(images)
        n,height,width,channels = images.shape
        process = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
                .output(fn, vcodec=vcodec, r=framerate)
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
    vidwrite(str(Path(path) / 'latent_error.mkv'), frames)
