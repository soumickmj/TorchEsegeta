import os
import sys
import logging
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
import nibabel as nb

from visualizations import create_max_projection
from captum.metrics import sensitivity_max, infidelity
from captum.metrics import infidelity_perturb_func_decorator
from utils import load_checkpoint, get_logger




@infidelity_perturb_func_decorator(False)
def perturb_fn_infidelity(inputs):
    # device = "cuda" if inputs.is_cuda else "cpu"
    noise = torch.tensor(np.random.normal(0, 0.0001, inputs.shape)).float().to(inputs.device)

    return inputs - noise


def perturb_fn_sensitivity(inputs):
    # device = "cuda" if inputs.is_cuda else "cpu"
    noise = torch.tensor(np.random.normal(0, 0.001, inputs.shape)).float().to(inputs.device)

    return (inputs - noise)


class Uncertainity:
    def __init__(self, **kwargs):

        self.__dict__.update(**kwargs)

    def infidelity_captum(self, forward_func, input_img, gradients, perturb_fn=perturb_fn_infidelity,
                          n_perturb_samples=1):

        return infidelity(forward_func=forward_func, inputs=input_img, attributions=gradients, perturb_func=perturb_fn,
                          n_perturb_samples=n_perturb_samples)

    def sensitivity_captum(self, input_img, method, target, device, library, perturb_fn=perturb_fn_sensitivity, n_perturb_samples=1, **kwargs):

        if library == "CNN Visualization":
            return sensitivity_max(inputs=input_img, target_class=target, device = device ,explanation_func=method, perturb_func=perturb_fn,
                               n_perturb_samples=n_perturb_samples, **kwargs)
        else:
            return sensitivity_max(inputs=input_img, target=target, explanation_func=method, perturb_func=perturb_fn,
                               n_perturb_samples=n_perturb_samples, **kwargs)

    def get_metrics(self, ip, grads, target, **kwargs):

        # try:
        #     self.logger.info("Running Uncertainity")
        #     print("ip:",ip.shape)
        #     print("grad:",grads.shape)

            if isinstance(ip, np.ndarray): ip = torch.from_numpy(ip)
            if isinstance(grads, np.ndarray): grads = torch.from_numpy(grads)
            grads = grads.to(self.device)

            if len(grads.shape)< len(ip.shape):
                if 64 in list(grads.shape):
                    axis = list(grads.shape).index(64)
                    grads = torch.max(grads, dim=axis).values


                    if not self.batch_dim_present:
                        grads = torch.unsqueeze(torch.unsqueeze(grads, 0), 0)
                    else:
                        ip = torch.squeeze(ip)
            #     if list(grads.shape) == list(ip.shape)[-2:]:
            #         grads = torch.unsqueeze(grads,0)
            #         grads = torch.unsqueeze(torch.cat((grads,grads,grads),0),0)
            #         grads.requires_grad = True
            # print(grads.shape)
            # print(ip.shape)

            infi = self.infidelity_captum(forward_func=self.forward_func, input_img=ip, gradients=grads,
                                          n_perturb_samples=1)

            # print("\nInfidelity for Saliency: ", infi.item())
            #
            sensi = self.sensitivity_captum(input_img=ip, target=target, method=self.method_attribution,
                                            n_perturb_samples=2, device= self.device, library= self.library, **kwargs)
            # print("\n Sensitivity for Saliency: ", sensi.item())

            return {"infidelity": infi.item(), "sensitivity": sensi.item()}

        # except:
        #     self.logger.error("Unexpected error")
        #     self.logger.error(sys.exc_info()[1])


def cascading_randomization(intr, method, forward_function, target, inp_image, inp_transform_flag,
                            transform_func, uncertainity_mode, visualize_method, sign, show_colorbar,
                            title, device, library, output_path, input_file,
                                                 is_3d , isDepthFirst, patcher_flag,
                                                 batch_dim_present, **kwargs):



    forward_func = copy.deepcopy(forward_function)
    layers = list(forward_func.children())
    if len(layers) == 1:
        layers = [l for l in layers[0].children()]

    layers.reverse()
    idx = 1
    uncertainity_metrics = []
    for layer in layers:

        for name, param in (layer.named_parameters()):
            forward_func.eval()
            std = torch.std_mean(param)[0].item()
            avg = torch.std_mean(param)[1].item()
            if np.isnan(std): std = 0
            if np.isnan(avg): avg = 0
            torch.nn.init.uniform_(param, avg - std, avg + std)


        forward_func.eval()

        if "nt_type" in kwargs.keys():
            kwargs["nt_type"] = False
        uncertain_metric = getattr(intr, method)(forward_func, target, inp_image, inp_transform_flag,
                    transform_func, True if uncertainity_mode ==2 else False, visualize_method, sign, show_colorbar, title, device,
                    library, output_path, input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present,
                     **kwargs, name_tag ="_CasRand"  + "_" + str(idx))


        uncertainity_metrics.append(uncertain_metric)
        idx += 1

    return uncertainity_metrics









class Uncertainity1:

    def __init__(self, interpretability_object, **kwargs):
        self.interpretability_object = interpretability_object
        self.__dict__.update(**kwargs)

        # , target, inp_image, Method, patcher_func, visualize_method,
        # sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
        # isDepthFirst, patcher_flag, abs, affine_size

    # def cascading_randomization(self, forward_function, method_name):
    #
    #     forward_func = copy.deepcopy(forward_function)
    #     layers = list(forward_func.children())
    #     if len(layers) == 1:
    #         layers = [l for l in layers[0].children()]
    #
    #     layers.reverse()
    #     fig = plt.figure(figsize=(13, 13))
    #     idx = 1
    #
    #     for layer in layers:
    #
    #         for name, param in (layer.named_parameters()):
    #             forward_func.eval()
    #             std = torch.std_mean(param)[0].item()
    #             avg = torch.std_mean(param)[1].item()
    #             if np.isnan(std): std = 0
    #             if np.isnan(avg): avg = 0
    #             torch.nn.init.uniform_(param, avg - std, avg + std)
    #
    #         # print(forward_func.state_dict()["base.Conv.weight"])
    #         forward_func.eval()
    #
    #         uncertain_metrics= getattr(self.interpretability_object, method_name)(forward_func, self.target, self.inp_image, self.inp_transform_flag,
    #                     self.transform_func, True if self.cacmode==2 else False, self.visualize_method, self.sign, self.show_colorbar, self.title, self.device,
    #                     self.library, self.output_path, self.input_file, self.is_3d, self.isDepthFirst, self.patcher_flag, self.batch_dim_present,
    #                     self.abs, self.nt_type, self.n_samples, self.affine_size, name_tag ="_CasRand" + str(idx) + "_")
    #         uncertain_metrics["method"] += "_CasRand" + str(idx)



    # def cascading_randomization(self, forward_function, target, inp_image, Method, patcher_func, visualize_method,
    #                             sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
    #                             isDepthFirst, patcher_flag, abs, affine_size):
    #
    #     forward_func = copy.deepcopy(forward_function)
    #     # print(forward_func.state_dict()["base.Conv.weight"])
    #     layers = list(forward_func.children())
    #     if len(layers) == 1:
    #         layers = [l for l in layers[0].children()]
    #
    #     layers.reverse()
    #     fig = plt.figure(figsize=(13, 13))
    #     idx = 1
    #
    #     for layer in layers:
    #
    #         for name, param in (layer.named_parameters()):
    #             forward_func.eval()
    #             std = torch.std_mean(param)[0].item()
    #             avg = torch.std_mean(param)[1].item()
    #             if np.isnan(std): std = 0
    #             if np.isnan(avg): avg = 0
    #             torch.nn.init.uniform_(param, avg - std, avg + std)
    #
    #         # print(forward_func.state_dict()["base.Conv.weight"])
    #         forward_func.eval()
    #         inp_image.requires_grad = True
    #
    #         if library == "captum":
    #             method = Method(forward_func)
    #             if patcher_flag:
    #                 grads, _ = patcher_func(method_handler=method.attribute,
    #                                      params={"ip": inp_image, "target": int(target), "abs": abs, "gpu_device": device})
    #             else:
    #
    #                 grads = method.attribute(inp_image, target=int(target), abs=abs)
    #
    #         if library == "CNN Visualization":
    #             pass
    #
    #         if is_3d:
    #             grads = grads.squeeze().cpu().detach().numpy()
    #             if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
    #
    #
    #
    #             create_max_projection(grads,
    #                                   np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
    #                                   else inp_image.detach().squeeze().to('cpu').numpy(),
    #                                   output_path, input_file + '_Saliency_Class_' + str(target) + str(idx))
    #
    #             img = nb.Nifti1Image(grads, np.eye(affine_size))
    #             nb.save(img,
    #                     os.path.join(output_path, input_file + '_Saliency_Class_' + str(target) + str(idx) + ".nii.gz"))
    #         else:
    #             if isDepthFirst:
    #                 grads = np.moveaxis(grads.squeeze().cpu().detach().numpy(), 0, -1)
    #             else:
    #                 grads = grads.squeeze().cpu().detach().numpy()
    #             fig.add_subplot(len(layers) // 2 + round(len(layers) % 2), 2, idx)
    #             plt.imshow(grads)
    #             original_image = np.transpose(inp_image.squeeze().cpu().detach().numpy(), (1, 2, 0))
    #             # axis = fig.add_subplot(1, len(layers), idx)
    #             # axis = fig.add_subplot(len(layers) // 2 + round(len(layers) % 2), 2, idx)
    #             # print(target)
    #             # fig, _ = viz.visualize_image_attr(grads, original_image, use_pyplot=False, method=visualize_method,
    #             #                                   sign=sign,
    #             #                                   show_colorbar=show_colorbar, title='Class ' + str(target) ,
    #             #                                   plt_fig_axis=(fig, axis))
    #         idx += 1
    #     if is_3d:
    #         return
    #     fig.suptitle(title)
    #     fig.savefig(output_path + "/" + input_file + str(Method) + 'uncertainity' + ".png")



