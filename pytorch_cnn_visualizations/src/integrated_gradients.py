"""
Created on Wed Jun 19 17:06:48 2019

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
import numpy as np

from pytorch_cnn_visualizations.src.misc_functions import get_example_params, convert_to_grayscale, save_gradient_images


class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model, is_3d, firstLayer = None):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers(is_3d, firstLayer)

    def hook_layers(self, is_3d, firstLayer=None):
        def hook_function(module, grad_in, grad_out):
            grad_in = torch.mean(grad_in[0], dim=0)
            self.gradients = grad_in

        # Register hook to the first layer
        if firstLayer is not None:
            first_layer = firstLayer
        else:
            if is_3d:
                first_layer = list(self.model._modules.items())[0][1][0]
            else:
                first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps+1)/steps
        return [input_image*step for step in step_list]

    def generate_gradients(self, input_image, target_class, device):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output.to(device), retain_graph = True)
        return self.gradients.data.cpu().numpy()[0]

    def generate_integrated_gradients(self, input_image, target_class, steps, device):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        # Initialize an iamge composed of zeros
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list:
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image.to(device), target_class, device)
            # Add rescaled grads from xbar images
            integrated_grads = integrated_grads + single_integrated_grad/steps
        # [0] to get rid of the first channel (1,3,224,224)
        return integrated_grads[0]


if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Vanilla backprop
    IG = IntegratedGradients(pretrained_model)
    # Generate gradients
    integrated_grads = IG.generate_integrated_gradients(prep_img, target_class, 100)
    # Convert to grayscale
    grayscale_integrated_grads = convert_to_grayscale(integrated_grads)
    # Save grayscale gradients
    save_gradient_images(
        grayscale_integrated_grads, f'{file_name_to_export}_Integrated_G_gray'
    )

    print('Integrated gradients completed.')
