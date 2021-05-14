import os
import numpy as np
from matplotlib import pyplot as plt


def create_max_projection(grad, image, output_path, method_name ):
    z_proj_input = np.max(image, axis=-1)
    # z_proj_input = np.clip(z_proj_input, 0, 1)
    result = np.max(grad, axis=-1)
    # result = np.clip(result,0, 1)

    # Saving attribution z project
    plt.imshow(result.transpose(), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(output_path , method_name + "_Z_prject.png"))

    # Saving attribution with overlay z project -- Tyoe1
    plt.imshow(z_proj_input.transpose(), cmap='gray')
    plt.imshow(result.transpose(), cmap='gist_heat', alpha=0.7)
    plt.axis('off')
    plt.savefig(os.path.join(output_path , method_name + "_Z_prject_overlay1.png"))

    # Saving attribution with overlay z project -- Tyoe1
    plt.imshow(z_proj_input.transpose(), cmap='Greys')
    plt.imshow(result.transpose(), cmap='Reds', alpha=0.7)
    plt.axis('off')
    plt.savefig(os.path.join(output_path , method_name + "_Z_prject_overlay2.png"))

