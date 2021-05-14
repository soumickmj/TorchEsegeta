"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch

from pytorch_cnn_visualizations.src.misc_functions import get_example_params, convert_to_grayscale, save_gradient_images


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, is_3d, firstLayer = None):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers(is_3d, firstLayer)

    def hook_layers(self,is_3d, firstLayer=None):
        def hook_function(module, grad_in, grad_out):

            #grad_in = torch.mean(grad_in[0], dim = 0)

            self.gradients = grad_in[0]
            if len(self.gradients.shape) > 4 and self.gradients.shape[1] > 3:
                self.gradients = torch.mean(self.gradients, dim=1)

        # Register hook to the first layer
        if firstLayer is not None:
            first_layer = firstLayer

        else:

            if is_3d: first_layer = list(self.model._modules.items())[0][1][0]
            else: first_layer = list(self.model._modules.items())[0][1]
        # print(first_layer)
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class, device):
        # Forward
        # print(input_image.shape)
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output.to(device), retain_graph=True)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)

        gradients_as_arr = self.gradients.data.cpu().numpy()[0]

        # print(gradients_as_arr.shape)

        return gradients_as_arr


if __name__ == '__main__':
    # Get params
    target_example = 1  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    print('Vanilla backprop completed')
