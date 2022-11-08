#########################################################################
## Project: Explain-ability and Interpret-ability for segmentation models
## Purpose: Ensemble python file for all functions present in thrid party libraries
## 			Methods extends functions from third party libraries
## Author: Arnab Das
#########################################################################

from captum.attr import Saliency,Occlusion,GuidedBackprop,IntegratedGradients,FeatureAblation,DeepLift,Deconvolution,\
  GuidedGradCam,LayerActivation,LayerConductance,LayerGradientShap, GradientShap, InternalInfluence, InputXGradient, DeepLiftShap, NoiseTunnel, \
  LayerGradientXActivation, LayerDeepLift, LayerGradCam, ShapleyValueSampling

from lucent.optvis import render

from pytorch_cnn_visualizations.src.guided_backprop import GuidedBackprop as Cnn_Vis_GBP
from pytorch_cnn_visualizations.src.integrated_gradients import IntegratedGradients as Cnn_Vis_IG
from pytorch_cnn_visualizations.src.gradcam import GradCam
from pytorch_cnn_visualizations.src.guided_gradcam import guided_grad_cam
from pytorch_cnn_visualizations.src.scorecam import ScoreCam
from pytorch_cnn_visualizations.src.vanilla_backprop import VanillaBackprop
from pytorch_cnn_visualizations.src.layer_activation_with_guided_backprop import GuidedBackprop as Cnn_Vis_LaGBP
from pytorch_cnn_visualizations.src.cnn_layer_visualization import CNNLayerVisualization
from pytorch_cnn_visualizations.src.deep_dream import DeepDream

from torchray.attribution.excitation_backprop import excitation_backprop, contrastive_excitation_backprop
from torchray.attribution.rise import rise
from torchray.attribution.deconvnet import deconvnet
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.gradient import gradient
from torchray.attribution.guided_backprop import guided_backprop
from torchray.attribution.linear_approx import linear_approx
from torchray.benchmark import plot_example
from visualizations import create_max_projection

from utils import normalize_gradient
import torch
from torch import nn
from captum.attr import visualization as viz
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import logging
import math
import nibabel as nb
import os
import torchio as tio
import pandas as pd
from torch.cuda.amp import autocast
from torchvision.transforms import ToTensor
import sys
from uncertainity import Uncertainity


class Interpretability:
    def __init__(self, model, available_wrapper_classes, patch_size=None, patch_overlap=None, amp_enabled = True, logger = logging.getLogger()):
        self.model = model
        self.wrapper_list = available_wrapper_classes
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.amp_enbled = amp_enabled
        self.logger = logger
        self.uncertain_metrics = []

    def save_uncertainity(self, path):
        df = pd.DataFrame(self.uncertain_metrics)
        df.to_csv(path)

    def patcher_func(self, method_handler, params, uncertainity=None):
        device = params["gpu_device"]
        params.pop("gpu_device")

        # print(params["ip"].shape)
        inp = tio.Subject(image=tio.ScalarImage(tensor=params["ip"].squeeze(0)))  # Squeeze to loose batch
        grid_sampler = tio.inference.GridSampler(inp, self.patch_size, self.patch_overlap, )
        patch_loader = torch.utils.data.DataLoader(grid_sampler,
                                                   batch_size=1)  # To be tested: batch_size as 1 and more. If works for more, make a param
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")

        params.pop("ip")
        uncertain_metrics_perpatch = []
        for indx, patches_batch in enumerate(patch_loader, start=1):
            input_tensor = patches_batch['image'][tio.DATA].float().to(device)
            locations = patches_batch[tio.LOCATION]
            with autocast(enabled=self.amp_enbled):
                ##attribution = method_handler(input_tensor, **params).detach().cpu()

                attribution = method_handler(input_tensor, **params)

                if uncertainity is not None:
                    target = params["target"] if "target" in params.keys() else params["target_class"]
                    uncertain_metrics_perpatch.append(
                        uncertainity.get_metrics(ip = input_tensor, target=target,grads= attribution))
                    # print(pd.DataFrame(uncertain_metrics_perpatch).dropna().median().to_dict())
                if type(attribution) is np.ndarray:
                    attribution = torch.tensor(attribution, dtype= torch.float32)
                else:
                    attribution = attribution.detach().cpu()
                    # torch.isnan(torch.max(attribution))
                    # np.isnan(np.max(np.array(attribution)))
                if np.isnan(np.max(np.array(attribution))) and self.amp_enbled:
                    with autocast(enabled=False):
                        attribution = method_handler(input_tensor, **params)
                        if type(attribution) is np.ndarray:
                            attribution = torch.tensor(attribution, dtype= torch.float32)
                        else:
                            attribution = attribution.detach().cpu()
            if list(attribution.shape) != list(input_tensor.shape):
                attribution = torch.reshape(attribution, list(input_tensor.shape))
            aggregator.add_batch(attribution, locations)
        output_tensor = aggregator.get_output_tensor()
        if uncertainity is not None:
            df = pd.DataFrame(uncertain_metrics_perpatch)
            # print("*********************************")
            # print(df)
            uncertain_metrics = df.dropna().median().to_dict()
            # print("*********************************")

        else:

            uncertain_metrics = None

        return output_tensor.unsqueeze(0), uncertain_metrics  # Unsqueeze to get back batch, for consist

    ## Input transformation function
    def inpTransformSetup(self, resize, centre_crop, mean_vec, std_vec):
        return transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(centre_crop),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_vec, std=std_vec),
            ]
        )

    ## Extended method for Captum saliency map
    def captum_Saliency(self, forward_func, target, inp_image, inp_transform_flag,
                        transform_func, uncertainity_flag, visualize_method, sign, show_colorbar, title, device,
                        library, output_path, input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present,
                        abs, nt_type, n_samples, affine_size, name_tag="_"):
        self.device = device


        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)
        ip.requires_grad = True
        forward_func.to(device).eval()

        saliency = Saliency(forward_func)
        fig = plt.figure(figsize=(9, 6))
        for idx, i in enumerate(target, start=1):

            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=Saliency,
                                            method_attribution=saliency.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)

            if patcher_flag:
                grads, uncertain_metrics = self.patcher_func(method_handler=saliency.attribute,
                                                             params={"ip": ip, "target": int(i), "abs": abs,
                                                                     "gpu_device": device},
                                                             uncertainity=uncertainity if uncertainity_flag else None)

            else:
                grads = saliency.attribute(ip, target=int(i), abs=abs)


                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=grads)


            if uncertainity_flag:

                uncertain_metrics["method"] = "captum_Saliency" + "_" +  str(i) + name_tag
                self.uncertain_metrics.append(uncertain_metrics)

            if is_3d:
                grads = grads.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)

                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_Saliency_Class_' + str(i) + name_tag)

                img = nb.Nifti1Image(grads, np.eye(affine_size))
                # print(img.shape)
                # print((np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1)).shape)
                nb.save(img, os.path.join(output_path, input_file + '_Saliency_Class_' + str(i) + name_tag+ ".nii.gz"))
            else:

                grads = (
                    np.moveaxis(grads.squeeze().cpu().detach().numpy(), 0, -1)
                    if isDepthFirst
                    else grads.squeeze().cpu().detach().numpy()
                )

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(
                    grads,
                    original_image,
                    use_pyplot=False,
                    method=visualize_method,
                    sign=sign,
                    show_colorbar=show_colorbar,
                    title=f'Class {str(i)}',
                    plt_fig_axis=(fig, axis),
                )

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Saliency' + visualize_method + name_tag+ ".png")
        if nt_type:
            nt = NoiseTunnel(saliency)
            fig = plt.figure(figsize=(9, 6))
            for idx, i in enumerate(target, start=1):
                grad_nt = nt.attribute(ip, nt_type=nt_type, n_samples=n_samples, target=int(i))
                grad_nt = np.transpose(grad_nt.squeeze().cpu().detach().numpy(), (1, 2, 0))
                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(
                    grad_nt,
                    original_image,
                    use_pyplot=False,
                    method=visualize_method,
                    sign=sign,
                    show_colorbar=show_colorbar,
                    title=f"Class {str(i)}",
                    plt_fig_axis=(fig, axis),
                )

            fig.suptitle(title + " with noise tunnel")
            fig.savefig(output_path + "/" + input_file + '_Saliency_noise_tunnel_' + visualize_method + ".png")

        ip.requires_grad = False
        if not uncertainity_flag:
            return  None
        metrics = self.uncertain_metrics
        self.uncertain_metrics = []
        return metrics


    def captum_Integrated_Gradients(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                    uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file,
                                    is_3d, isDepthFirst, patcher_flag,
                                    batch_dim_present, nt_type, n_samples, affine_size, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        ig = IntegratedGradients(forward_func)
        fig = plt.figure(figsize=(9, 6))
        for idx, i in enumerate(target, start=1):
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=Saliency,
                                            method_attribution=ig.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                grads, uncertain_metrics = self.patcher_func(method_handler=ig.attribute,
                                                             params={"ip": ip, "target": int(i),
                                                                     "gpu_device": device},
                                                             uncertainity=uncertainity if uncertainity_flag else None)
            else:
                grads = ig.attribute(ip, target=int(i))
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=grads)


            if uncertainity_flag:

                uncertain_metrics["method"] = "captum_IntegratedGradients" + "_" +  str(i) + name_tag

                self.uncertain_metrics.append(uncertain_metrics)


            if is_3d:
                grads = grads.squeeze().cpu().detach().numpy()
                if isDepthFirst: np.moveaxis(grads, 0, -1)

                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_IG_Class_' + str(i) + name_tag)

                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_IG_Class_' + str(i) +name_tag+ ".nii.gz"))
            else:
                grads = (
                    np.moveaxis(grads.squeeze().cpu().detach().numpy(), 0, -1)
                    if isDepthFirst
                    else grads.squeeze().cpu().detach().numpy()
                )

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(
                    grads,
                    original_image,
                    use_pyplot=False,
                    method=visualize_method,
                    sign=sign,
                    show_colorbar=show_colorbar,
                    title=f'Class {str(i)}',
                    plt_fig_axis=(fig, axis),
                )

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics

            return
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_IG_' + visualize_method + name_tag + ".png")

        if nt_type:
            nt = NoiseTunnel(ig)
            fig = plt.figure(figsize=(9, 6))
            for idx, i in enumerate(target, start=1):
                grads_nt = nt.attribute(ip, nt_type=nt_type, n_samples=n_samples, target=int(i))
                grads_nt = np.transpose(grads_nt.squeeze().cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(
                    grads_nt,
                    original_image,
                    use_pyplot=False,
                    method=visualize_method,
                    sign=sign,
                    show_colorbar=show_colorbar,
                    title=f"Class {str(i)}",
                    plt_fig_axis=(fig, axis),
                )

            fig.suptitle(title + " with noise tunnel")
            fig.savefig(output_path + "/" + input_file + '_IG_noise_tunnel_' + visualize_method + ".png")

        ip.requires_grad = False

        if not uncertainity_flag:
            return None
        metrics = self.uncertain_metrics
        self.uncertain_metrics = []
        return metrics

    def captum_Feature_Ablation(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                uncertainity_flag,visualize_method, sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
                                isDepthFirst,
                                patcher_flag, batch_dim_present, perturbations_per_eval, nt_type, n_samples, mask=None,
                                affine_size=4, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        self.model.to(device).eval()

        fig1 = plt.figure(figsize=(9, 6))
        fig2 = plt.figure(figsize=(9, 6))

        idx = 1

        fa = FeatureAblation(forward_func)
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=FeatureAblation,
                                            method_attribution=fa.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                fa_attr, uncertain_metrics = self.patcher_func(method_handler=fa.attribute,
                                                             params={"ip": ip, "target": int(i),
                                                                     "gpu_device": device},
                                                             uncertainity=uncertainity if uncertainity_flag else None)
            else:
                fa_attr = fa.attribute(ip, target=int(i))
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=fa_attr)

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_FeatureAblation" + "_" +  str(i) + name_tag

                self.uncertain_metrics.append(uncertain_metrics)

            if is_3d:
                grads = fa_attr.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_FA_withoutMask_Class_' + str(i) + name_tag + ".nii.gz"))
            else:
                ax1 = fig1.add_subplot(1, len(target), idx)
                if isDepthFirst:
                    fig1, _ = viz.visualize_image_attr(fa_attr[0].cpu().detach().permute(1, 2, 0).numpy(),
                                                       use_pyplot=False, sign=sign, show_colorbar=show_colorbar,
                                                       method=visualize_method,
                                                       title="Class " + str(int(i)) , plt_fig_axis=(fig1, ax1))
                else:
                    fig1, _ = viz.visualize_image_attr(fa_attr[0].cpu().detach().numpy(), use_pyplot=False, sign=sign,
                                                       show_colorbar=show_colorbar, method=visualize_method,
                                                       title="Class " + str(int(i))  , plt_fig_axis=(fig1, ax1))

            if mask is not None:
                fa_attr_mask = fa.attribute(ip, feature_mask=mask, perturbations_per_eval=perturbations_per_eval,
                                            target=int(i))
                if is_3d:
                    grads = fa_attr_mask.squeeze().cpu().detach().numpy()
                    if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                    img = nb.Nifti1Image(grads, np.eye(affine_size))
                    nb.save(img, os.path.join(output_path, input_file + '_FA_Without_Max_Class_' + str(i)+ name_tag + ".nii.gz"))
                else:
                    ax2 = fig2.add_subplot(1, len(target), idx)
                    if isDepthFirst:
                        fig2, _ = viz.visualize_image_attr(fa_attr_mask[0].cpu().detach().permute(1, 2, 0).numpy(),
                                                           use_pyplot=False, method=visualize_method, sign=sign,
                                                           show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                                           plt_fig_axis=(fig2, ax2))
                    else:
                        fig2, _ = viz.visualize_image_attr(fa_attr_mask[0].cpu().detach().numpy(), use_pyplot=False,
                                                           method=visualize_method, sign=sign,
                                                           show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                                           plt_fig_axis=(fig2, ax2))
            idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        fig1.suptitle(title)
        fig1.savefig(output_path + "/" + input_file + '_FA_withoutMask_' + visualize_method + name_tag + ".png")
        if mask is not None:
            fig2.suptitle(title + " without max")
            fig2.savefig(output_path + "/" + input_file + '_FA_wth_mask_' + visualize_method + name_tag+ ".png")

        if nt_type:
            nt = NoiseTunnel(fa)
            idx = 1
            fig1 = plt.figure(figsize=(9, 6))
            for i in target:
                fa_attr_nt = nt.attribute(ip, nt_type=nt_type, n_samples=n_samples,
                                          perturbations_per_eval=perturbations_per_eval,
                                          target=int(i))
                if isDepthFirst:
                    fa_attr_nt = np.transpose(fa_attr_nt.squeeze().cpu().detach().numpy(), (1, 2, 0))
                else:
                    fa_attr_nt = fa_attr_nt.squeeze().cpu().detach().numpy()
                ax1 = fig1.add_subplot(1, len(target), idx)
                fig1, _ = viz.visualize_image_attr(fa_attr_nt, use_pyplot=False, method=visualize_method,
                                                   sign=sign, show_colorbar=show_colorbar,
                                                   title="Class " + str(int(i)), plt_fig_axis=(fig1, ax1))
                idx += 1
            fig1.suptitle(title + " with noise tunnel")
            fig1.savefig(output_path + "/" + input_file + '_FA_noise_tunnel_' + visualize_method + ".png")

        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def captum_occlusion(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                         uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
                         isDepthFirst, patcher_flag,
                         batch_dim_present, sliding_window_shapes, strides, nt_type, n_samples, affine_size=4, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        oc = Occlusion(forward_func)
        fig = plt.figure(figsize=(9, 6))
        idx = 1
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=Occlusion,
                                            method_attribution=oc.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                attr_oc, uncertain_metrics = self.patcher_func(method_handler=oc.attribute,
                                    params={"ip": ip,
                                            "sliding_window_shapes": tuple(map(int, sliding_window_shapes.split(","))),
                                            "target": int(i), "strides": strides, "gpu_device": device}
,
                                                               uncertainity=uncertainity if uncertainity_flag else None)
            else:
                attr_oc = oc.attribute(ip, target=int(i), sliding_window_shapes=tuple(map(int, sliding_window_shapes.split(","))), strides=strides)
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=attr_oc, sliding_window_shapes = tuple(map(int, sliding_window_shapes.split(","))))

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_Occlusion" + "_" +  str(i) + name_tag

                self.uncertain_metrics.append(uncertain_metrics)

            if is_3d:
                grads = attr_oc.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Occlusion_' + str(i) +name_tag +".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_Occlusion_' + str(i) + name_tag)
            else:
                if isDepthFirst:
                    attr_oc = np.moveaxis(attr_oc.squeeze().cpu().detach().numpy(), 0, -1)

                else:
                    attr_oc = attr_oc.squeeze(0).cpu().detach().numpy()

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attr_oc, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

                #plt.subplot(1, len(target), idx)
                #plt.imshow(original_image, cmap='Greys')
                #plt.imshow(attr_oc, cmap='Reds', alpha=0.5)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

            idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Occlusion_' + visualize_method + name_tag + ".png")

        if nt_type:
            nt_oc = NoiseTunnel(oc)
            fig = plt.figure(figsize=(9, 6))
            idx = 1
            for i in target:
                nt_attr_oc = nt_oc.attribute(ip, nt_type=nt_type, n_samples=n_samples, target=int(i),
                                             sliding_window_shapes=tuple(map(int, sliding_window_shapes.split(","))),
                                             strides=strides)
                nt_attr_oc = np.transpose(nt_attr_oc.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(nt_attr_oc, original_image, use_pyplot=False, sign=sign,
                                                  method=visualize_method,
                                                  show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                                  plt_fig_axis=(fig, axis))
                idx += 1

            fig.suptitle(title + " with noise tunnel")
            fig.savefig(output_path + "/" + input_file + '_Occlusion_noise_tunnel_' + visualize_method + ".png")
            ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def captum_guided_backprop(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                               uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
                               isDepthFirst,
                               patcher_flag, batch_dim_present, nt_type, n_samples, affine_size, name_tag = "_"):

        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        gbp = GuidedBackprop(forward_func)
        fig = plt.figure(figsize=(9, 6))
        idx = 1
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=GuidedBackprop,
                                            method_attribution=gbp.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                attr_gbp, uncertain_metrics = self.patcher_func(method_handler=gbp.attribute,
                                                               params={"ip": ip, "target": int(i),
                                                                       "gpu_device": device},
                                                               uncertainity=uncertainity if uncertainity_flag else None)
            else:
                attr_gbp = gbp.attribute(ip, target=int(i))
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=attr_gbp)

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_guided_backprop" + "_" +  str(i) + name_tag

                self.uncertain_metrics.append(uncertain_metrics)
            if is_3d:
                grads = attr_gbp.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)

                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_GBP_Class_' + str(i) + name_tag)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_GBP_Class_' + str(i) + name_tag+ ".nii.gz"))
            else:
                if isDepthFirst:
                    attr_gbp = np.moveaxis(attr_gbp.squeeze().cpu().detach().numpy(), 0, -1)
                else:
                    attr_gbp = attr_gbp.squeeze().cpu().detach().numpy()

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attr_gbp, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

        idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_GBP_' + visualize_method +name_tag+ ".png")

        if nt_type:
            nt_gbp = NoiseTunnel(gbp)
            fig = plt.figure(figsize=(9, 6))
            idx = 1
            for i in target:
                nt_attr_gbp = nt_gbp.attribute(ip, nt_type=nt_type, n_samples=n_samples, target=int(i))
                nt_attr_gbp = np.transpose(nt_attr_gbp.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(nt_attr_gbp, original_image, use_pyplot=False, sign=sign,
                                                  method=visualize_method,
                                                  show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                                  plt_fig_axis=(fig, axis))
                idx += 1

            fig.suptitle(title + " with noise tunnel")
            fig.savefig(output_path + "/" + input_file + '_GBP_noise_tunnel_' + visualize_method + ".png")
        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def captum_deep_lift(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                         uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
                         isDepthFirst,
                         patcher_flag, batch_dim_present, nt_type, n_samples, affine_size, name_tag = "_"):

        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        dl = DeepLift(forward_func)
        fig = plt.figure(figsize=(9, 6))
        idx = 1
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=DeepLift,
                                            method_attribution=dl.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                attr_dl, uncertain_metrics = self.patcher_func(method_handler=dl.attribute,
                                                                params={"ip": ip, "target": int(i),
                                                                        "gpu_device": device},
                                                                uncertainity=uncertainity if uncertainity_flag else None)
            else:
                attr_dl = dl.attribute(ip, target=int(i))
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=attr_dl)

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_deep_lift"  + "_" +  str(i) + name_tag

                self.uncertain_metrics.append(uncertain_metrics)

            if is_3d:
                grads = attr_dl.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)

                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_DLift_Class_' + str(i)+ name_tag)

                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_DLift_Class_' + str(i) + name_tag+".nii.gz"))
            else:
                if isDepthFirst:
                    attr_dl = np.moveaxis(attr_dl.squeeze().cpu().detach().numpy(), 0, -1)
                else:
                    attr_dl = attr_dl.squeeze().cpu().detach().numpy()

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attr_dl, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

        idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_DLift_' + visualize_method + name_tag+ ".png")

        if nt_type:
            nt_dl = NoiseTunnel(dl)
            fig = plt.figure(figsize=(9, 6))
            idx = 1
            for i in target:
                nt_attr_dl = nt_dl.attribute(ip, nt_type=nt_type, n_samples=n_samples, target=int(i))
                nt_attr_dl = np.transpose(nt_attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(nt_attr_dl, original_image, use_pyplot=False, sign=sign,
                                                  method=visualize_method,
                                                  show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                                  plt_fig_axis=(fig, axis))
                idx += 1

            fig.suptitle(title + " with noise tunnel")
            fig.savefig(output_path + "/" + input_file + '_Dlift_noise_tunnel_' + visualize_method + ".png",
                        )
        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def captum_deconvolution(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                             uncertainity_flag ,visualize_method, sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
                             isDepthFirst,
                             patcher_flag, batch_dim_present, nt_type, n_samples, affine_size , name_tag = "_"):

        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        deconv = Deconvolution(forward_func)
        fig = plt.figure(figsize=(9, 6))
        idx = 1
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=Deconvolution,
                                            method_attribution=deconv.attribute,
                                            patcher_func=self.patcher_func, device=device,library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                attr_dconv, uncertain_metrics = self.patcher_func(method_handler=deconv.attribute,
                                                                  params={"ip": ip, "target": int(i),
                                                                          "gpu_device": device},
                                                                  uncertainity=uncertainity if uncertainity_flag else None)
            else:
                attr_dconv = deconv.attribute(ip, target=int(i))
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=attr_dconv)

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_deconvolution" + "_" +  str(i) + name_tag
                self.uncertain_metrics.append(uncertain_metrics)
            if is_3d:
                grads = attr_dconv.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)

                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_Deconv_Class_' + str(i)+ name_tag)

                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Deconv_Class_' + str(i) +name_tag+ ".nii.gz"))
            else:
                if isDepthFirst:
                    attr_dconv = np.moveaxis(attr_dconv.squeeze().cpu().detach().numpy(), 0, -1)
                else:
                    attr_dconv = attr_dconv.squeeze().cpu().detach().numpy()

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attr_dconv, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Deconv_' + visualize_method + name_tag+ ".png", format='png')

        if nt_type:
            nt_dconv = NoiseTunnel(deconv)
            fig = plt.figure(figsize=(9, 6))
            idx = 1
            for i in target:
                nt_attr_dconv = nt_dconv.attribute(ip, nt_type=nt_type, n_samples=n_samples, target=int(i))
                nt_attr_dconv = np.transpose(nt_attr_dconv.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(nt_attr_dconv, original_image, use_pyplot=False, sign=sign,
                                                  method=visualize_method,
                                                  show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                                  plt_fig_axis=(fig, axis))
                idx += 1

            fig.suptitle(title + " with noise tunnel")
            fig.savefig(output_path + "/" + input_file + '_Deconv_noise_tunnel_' + visualize_method + ".png",
                        format='png')
        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def captum_guided_grad_cam(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                               uncertainity_flag ,visualize_method, sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
                               isDepthFirst,
                               patcher_flag, batch_dim_present, layer, nt_type, n_samples, affine_size, name_tag = "_"):

        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        modelName = [k for k, v in locals().items() if v == forward_func][0]

        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base.' + layer

        ggcam = GuidedGradCam(forward_func, eval(modelName + "." + layer))
        fig = plt.figure(figsize=(9, 6))
        idx = 1
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=GuidedGradCam,
                                            method_attribution=ggcam.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                attr_ggcam, uncertain_metrics = self.patcher_func(method_handler=ggcam.attribute,
                                                               params={"ip": ip, "target": int(i), "gpu_device": device},
                                                               uncertainity=uncertainity if uncertainity_flag else None)
            else:
                attr_ggcam = ggcam.attribute(ip, target=int(i))
                # print(attr_ggcam.shape)
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=attr_ggcam)

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_guided_grad_cam" + "_" +  str(i) + name_tag
                self.uncertain_metrics.append(uncertain_metrics)
            if is_3d:
                grads = attr_ggcam.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_Guided_GCam_' + str(i)+ name_tag)

                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Guided_GCam_' + str(i) + name_tag+ ".nii.gz"))
            else:
                if isDepthFirst:
                    attr_ggcam = np.moveaxis(attr_ggcam.squeeze().cpu().detach().numpy(), 0, -1)
                else:
                    attr_ggcam = attr_ggcam.squeeze().cpu().detach().numpy()

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attr_ggcam, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Guided_GCam_' + visualize_method + name_tag+ ".png", format='png')

        if nt_type:
            nt_ggcam = NoiseTunnel(ggcam)
            fig = plt.figure(figsize=(9, 6))
            idx = 1
            for i in target:
                nt_attr_ggcam = nt_ggcam.attribute(ip, nt_type=nt_type, n_samples=n_samples, target=int(i))
                nt_attr_ggcam = np.transpose(nt_attr_ggcam.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(nt_attr_ggcam, original_image, use_pyplot=False, sign=sign,
                                                  method=visualize_method,
                                                  show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                                  plt_fig_axis=(fig, axis))
                idx += 1

            fig.suptitle(title + " with noise tunnel")
            fig.savefig(output_path + "/" + input_file + '_Guided_GCam_noise_tunnel_' + visualize_method + ".png",
                        format='png')

        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None



    def captum_layer_activaltion(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                 uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
                                 isDepthFirst,
                                 patcher_flag, batch_dim_present, layer, neuron_start, neuron_end, affine_size=4, name_tag = "_"):
        if neuron_end < neuron_start:
            self.logger.error("Improper index of neurons")
            self.logger.error("neuron_end count must not be less than neuron_start")
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        modelName = [k for k, v in locals().items() if v == forward_func][0]
        ip.requires_grad = True
        forward_func.to(device).eval()

        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base.' + layer

        la = LayerActivation(forward_func, eval(modelName + "." + layer))
        if patcher_flag:
            attr_la = self.patcher_func(method_handler=la.attribute, params={"ip": ip, "gpu_device": device})
        else:
            attr_la = la.attribute(ip)
        if isDepthFirst:
            attr_la = np.moveaxis(attr_la.squeeze().cpu().detach().numpy(), 0, -1)
        else:
            attr_la = attr_la.squeeze().cpu().detach().numpy()

        if is_3d:
            for idx in range(neuron_start, neuron_end):
                grads = attr_la[..., idx]
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Layer_Activation_Maps_' + str(idx) + name_tag+ ".nii.gz"))
            return
        total_feature_maps = neuron_end - neuron_start
        plt.figure(figsize=(30, 30))
        plt.suptitle(title, y=.8)
        for idx in range(neuron_start, neuron_end):
            plt.subplot(math.floor(total_feature_maps / 5) + 1, 5, idx - (neuron_start - 1))
            plt.imshow(attr_la[:, :, idx], cmap="binary")
            plt.xticks([])
            plt.yticks([])
            #plt.colorbar()
            plt.xlabel("activation" + str(idx), fontsize=18)
        plt.savefig(output_path + "/" + input_file + '_Layer_Activation_Maps_' + name_tag+ ".png", format='png')

    def cnn_vis_guided_backprop(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
                                isDepthFirst, patcher_flag, batch_dim_present, firstLayer, affine_size, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        if len(firstLayer) > 0:
            if type(forward_func).__name__ in self.wrapper_list:
                firstLayer = 'base.' + firstLayer
            modelName = [k for k, v in locals().items() if v == forward_func][0]
            firstLayer = eval(modelName + "." + firstLayer)
            gbp = Cnn_Vis_GBP(forward_func, is_3d, firstLayer)
        else:
            gbp = Cnn_Vis_GBP(forward_func, is_3d)
        fig = plt.figure(figsize=(5, 5))
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        idx = 1
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=Cnn_Vis_IG,
                                            method_attribution=gbp.generate_gradients,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                grads, uncertain_metrics = self.patcher_func(method_handler=gbp.generate_gradients,
                                          params={"ip": ip, "target_class": int(i), "gpu_device": device,
                                                  "device": device},
                                                                  uncertainity=uncertainity if uncertainity_flag else None)
            else:
                grads = gbp.generate_gradients(ip, int(i), device)
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=grads)

            if uncertainity_flag:
                uncertain_metrics["method"] = "cnn_vis_guided_backprop" + "_" +  str(i) + name_tag

            if is_3d:
                grads = grads.squeeze()
                if type(grads) is not np.ndarray:
                    grads = grads.numpy().astype('float32')
                if isDepthFirst:
                    if len(grads.shape) > 3:
                        grads = np.moveaxis(grads, [0, 1], [-1, -2])
                    else:
                        grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_CNNVis_Guided_BackProp_' + str(i) + name_tag+ ".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_CNNVis_Guided_BackProp_' + str(i) + name_tag)
            else:
                if type(grads) is not np.ndarray:
                    grads = grads.numpy().astype('float32')
                if isDepthFirst:
                    grads = np.transpose(grads.astype('float32'), (1, 2, 0))

                #if len(grads.shape) == 3: grads *= 255
                #plt.subplot(1, len(target), idx)
                #plt.imshow(grads, cmap='Greys')
                #plt.xticks([])
                #plt.yticks([])
                #plt.colorbar()
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(grads, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))



            idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_CNNVis_Guided_BackProp' +name_tag+ ".png", format='png')
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_CNNVis_Guided_BackProp_' + visualize_method + name_tag + ".png",
                    format='png')

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def cnn_vis_integrated_grad(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                uncertainity_flag ,visualize_method, sign, show_colorbar, title, device, library, output_path,
                                input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, n_samples, firstLayer,
                                affine_size, name_tag = "_"):


        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        if len(firstLayer) > 0:
            modelName = [k for k, v in locals().items() if v == forward_func][0]
            if type(forward_func).__name__ in self.wrapper_list:
                firstLayer = 'base.' + firstLayer
            firstLayer = eval(modelName + "." + firstLayer)
            ig = Cnn_Vis_IG(forward_func, is_3d, firstLayer)
        else:
            ig = Cnn_Vis_IG(forward_func, is_3d)
        fig = plt.figure(figsize=(5, 5))
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        idx = 1
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=Cnn_Vis_IG,
                                            method_attribution=ig.generate_integrated_gradients,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                inti_grads, uncertain_metrics = self.patcher_func(method_handler=ig.generate_integrated_gradients,
                                                                  params={"ip": ip, "target_class": int(i), "steps": n_samples,
                                                                          "gpu_device": device, "device": device},
                                                                  uncertainity=uncertainity if uncertainity_flag else None)
            else:
                inti_grads = ig.generate_integrated_gradients(ip, int(i), n_samples, device)
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics( ip=ip, target=int(i), grads=inti_grads, steps = int(n_samples))

            if uncertainity_flag:
                uncertain_metrics["method"] = "cnn_vis_integrated_grad" + "_" +  str(i) + name_tag

            if is_3d:
                grads = inti_grads.squeeze()
                if type(grads) is not np.ndarray:
                    grads = grads.numpy().astype('float32')
                if isDepthFirst:
                    if len(grads.shape) > 3:
                        grads = np.moveaxis(grads, [0, 1], [-1, -2])
                    else:
                        grads = np.moveaxis(grads, 0, -1)

                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_CNNVis_IG_' + str(i) + name_tag+ ".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_CNNVis_IG_' + str(i)+ name_tag)
            else:
                if type(inti_grads) is not np.ndarray:
                    grads = inti_grads.numpy().astype('float32')
                if isDepthFirst:
                    inti_grads = np.transpose(inti_grads, (1, 2, 0))
                # if len(inti_grads.shape) == 3: inti_grads *= 255

                #plt.subplot(1, len(target), idx)
                #plt.imshow(inti_grads, cmap='Greys')
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(inti_grads, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_CNNVis_IG_' +name_tag+ ".png", format='png')

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_CNNVis_IG_' + visualize_method + name_tag + ".png",
                    format='png')

        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def cnn_vis_guided_grad_cam(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path,
                                input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, layer, firstLayer,
                                affine_size, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base.' + layer
            firstLayer = 'base.' + firstLayer

        gcam = GradCam(forward_func, target_layer=layer)
        if len(firstLayer) > 0:
            modelName = [k for k, v in locals().items() if v == forward_func][0]
            firstLayer = eval(modelName + "." + firstLayer)
            gbp = Cnn_Vis_GBP(forward_func, is_3d, firstLayer)
        else:
            gbp = Cnn_Vis_GBP(forward_func, is_3d)
        fig = plt.figure(figsize=(5, 5))
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        idx = 1
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=Cnn_Vis_GBP,
                                            method_attribution=gcam.generate_cam,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                cam, _ = self.patcher_func(method_handler=gcam.generate_cam,
                                                                  params={"ip": ip, "device": device, "is_3d": is_3d, "target_class": int(i),
                                                                       "gpu_device": device},
                                                                  uncertainity=uncertainity if uncertainity_flag else None)
                gbp_grads, _ = self.patcher_func(method_handler=gbp.generate_gradients,
                                              params={"ip": ip, "gpu_device": device, "target_class": int(i),
                                                      "device": device},
                                                                  uncertainity=uncertainity if uncertainity_flag else None)


            else:
                cam = gcam.generate_cam(ip, device, is_3d, int(i))
                gbp_grads = gbp.generate_gradients(ip, int(i), device)


            gg_cam = guided_grad_cam(cam, gbp_grads)

            if uncertainity_flag:
                uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=gg_cam)
                uncertain_metrics["method"] = "cnn_vis_guided_grad_cam" + "_" +  str(i) + name_tag

            if is_3d:
                grads = normalize_gradient(gg_cam.squeeze())
                if type(grads) is not np.ndarray:
                    grads = grads.cpu().detach().numpy().astype('float32')
                if isDepthFirst:
                    if len(grads.shape) > 3:
                        grads = np.moveaxis(grads, [0, 1], [-1, -2])
                    else:
                        grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_CNNVis_Guided_Grad_Cam_' + str(i) + name_tag+ ".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_CNNVis_Guided_Grad_Cam_' + str(i)+ name_tag)
            else:
                if type(gg_cam) is not np.ndarray:
                    grads = gg_cam.cpu().detach().numpy().astype('float32')
                if isDepthFirst:
                    gg_cam = np.transpose(gg_cam, (1, 2, 0))
                else:
                    gg_cam = gg_cam

                #if len(gg_cam.shape) == 3: gg_cam *= 255
                #plt.subplot(1, len(target), idx)
                #plt.imshow(gg_cam)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(gg_cam, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_CNNVis_Guided_Grad_Cam_' +name_tag+ ".png", format='png')

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_CNNVis_Guided_Grad_Cam_' + visualize_method + name_tag + ".png",
                    format='png')

        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def cnn_vis_score_cam(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                          uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library,
                          output_path, input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, layer,
                          affine_size, name_tag = "_"):
        self.device = device
        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base.' + layer

        input_shape = list(ip.size())
        score_cam = ScoreCam(forward_func, target_layer=layer)
        plt.figure(figsize=(5, 5))
        idx = 1
        for i in target:
            if is_3d:
                if patcher_flag:
                    scam, _ = self.patcher_func(method_handler=score_cam.generate_cam,
                                                params={"ip": ip, "device": device, "size": (self.patch_size, self.patch_size, self.patch_size), "is_3d": is_3d,
                                                       "target_class": int(i),
                                                       "gpu_device": device},
                                                        uncertainity= None)
                else:
                    scam = score_cam.generate_cam(ip, device, (input_shape[-3], input_shape[-2], input_shape[-1]), is_3d,
                                              int(i))
                grads = normalize_gradient(scam.squeeze())

                if type(grads) is not np.ndarray:
                    grads = grads.cpu().detach().numpy().astype('float32')

                if isDepthFirst:
                    grads= np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_CNNVis_Score_Cam_' + str(i) + name_tag+".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_CNNVis_Score_Cam_' + str(i)+ name_tag)
            else:
                scam = score_cam.generate_cam(ip, device, (input_shape[-1], input_shape[-2]), is_3d, int(i))

                if type(scam) is not np.ndarray:
                    scam = scam.cpu().detach().numpy().astype('float32')

                plt.subplot(1, len(target), idx)
                plt.imshow(scam, cmap='Reds')
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()
                plt.title("Class " + str(int(i)))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        plt.suptitle(title)
        plt.savefig(output_path + "/" + input_file + '_CNNVis_Score_Cam_' +name_tag+ ".png", format='png')

        ip.requires_grad = False

    def captum_layer_conductance(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                 uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
                                 isDepthFirst,
                                 patcher_flag, batch_dim_present, layer, device_ids, affine_size, name_tag = "_"):

        if len(device_ids) > 1 and (device.split(':')[-1] != str(device_ids[0])):
            raise Exception(
                "For multi device execution your device id must match the fist device number of the device id list")
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        forward_func.to(device).eval()

        modelName = [k for k, v in locals().items() if v == forward_func][0]
        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base.' + layer

        if (len(device_ids) > 0):
            forward_func = nn.DataParallel(forward_func, device_ids)
            l_cond = LayerConductance(forward_func, eval(modelName + ".module." + layer), device_ids=device_ids)
        else:
            l_cond = LayerConductance(forward_func, eval(modelName + "." + layer))

        fig = plt.figure(figsize=(9, 6))
        idx = 1
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        for i in target:
            if patcher_flag:
                attr_lcond = self.patcher_func(method_handler=l_cond.attribute,
                                               params={"ip": ip, "target": int(i), "n_steps": 1, "gpu_device": device})
            else:
                attr_lcond = l_cond.attribute(ip, target=int(i), n_steps=1)
            if isDepthFirst:
                if len(attr_lcond.shape) > 3:
                    attr_lcond = np.moveaxis(attr_lcond.squeeze(0).cpu().detach().numpy(), [0, 1], [-1, -2])
                else:
                    attr_lcond = np.moveaxis(attr_lcond.squeeze(0).cpu().detach().numpy(), 0, -1)
            else:
                attr_lcond = attr_lcond.squeeze(0).cpu().detach().numpy()
            if is_3d:
                grads = attr_lcond.squeeze()
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Layer_Cond_' + str(i) + name_tag+ ".nii.gz"))
            else:

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attr_lcond, original_image, use_pyplot=False, sign=sign,
                                                  method=visualize_method,
                                                  show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                                  plt_fig_axis=(fig, axis))
            idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Layer_Cond_' + visualize_method +name_tag+ ".png", format='png')
        ip.requires_grad = False

    def cnn_vis_vanilla_backprop(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                 uncertainity_flag ,visualize_method, sign, show_colorbar, title, device, library,
                                 output_path, input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present,
                                 firstLayer = "Conv1" , affine_size = 4, name_tag = "_"):
        # print(inp_image.shape)
        self.device = device
        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True

        forward_func.to(device).eval()

        if len(firstLayer) > 0:
            modelName = [k for k, v in locals().items() if v == forward_func][0]
            if type(forward_func).__name__ in self.wrapper_list:
                firstLayer = 'base.' + firstLayer
            firstLayer = eval(modelName + "." + firstLayer)
            VBP = VanillaBackprop(forward_func, is_3d, firstLayer)
        else:
            VBP = VanillaBackprop(forward_func, is_3d)
        fig = plt.figure(figsize=(5, 5))
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        idx = 1
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=VanillaBackprop,
                                            method_attribution=VBP.generate_gradients,
                                            patcher_func=self.patcher_func, device=device, library = library, batch_dim_present = batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                vanilla_grads, uncertain_metrics = self.patcher_func(method_handler=VBP.generate_gradients,
                                                                  params={"ip": ip, "target_class": int(i),
                                                                          "gpu_device": device, "device": device},
                                                                  uncertainity=uncertainity if uncertainity_flag else None)
            else:
                vanilla_grads = VBP.generate_gradients(ip, int(i), device)
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=vanilla_grads)

            if uncertainity_flag:
                uncertain_metrics["method"] = "cnn_vis_vanilla_backprop" + "_" +  str(i) + name_tag


            if is_3d:
                grads = vanilla_grads.squeeze()
                if type(grads) is not np.ndarray:
                    grads = grads.cpu().detach().numpy().astype('float32')
                # print(grads.dtype)
                if isDepthFirst:
                    if len(grads.shape) > 3:
                        grads = np.moveaxis(grads, [0, 1], [-1, -2])
                    else:
                        grads = np.moveaxis(grads, 0, -1)
                # print(grads.shape)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                # print(img.shape)
                nb.save(img, os.path.join(output_path, input_file + '_CNNVis_Vanilla_BackProp_' + str(i) +name_tag+ ".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_CNNVis_Vanilla_BackProp_' + str(i)+ name_tag)
            else:
                if type(vanilla_grads) is not np.ndarray:
                    vanilla_grads = vanilla_grads.cpu().detach().numpy().astype('float32')
                if isDepthFirst:
                    vanilla_grads = np.transpose(vanilla_grads, (1, 2, 0))
                else:
                    vanilla_grads = vanilla_grads

                # if len(vanilla_grads.shape) == 3: vanilla_grads *= 255

                #if isDepthFirst:
                #    original = np.moveaxis(inp_image.squeeze().detach().cpu().numpy(), 0, -1) \
                #        if len(inp_image.squeeze().detach().cpu().numpy().shape) == 3 \
                #        else inp_image.squeeze().detach().cpu().numpy()
                #else:
                #    original = inp_image.detach().cpu().numpy()

                #plt.subplot(1, len(target), idx)
                #plt.imshow(original, cmap='Greys')
                #plt.imshow(vanilla_grads, cmap='Reds', alpha=0.5)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(vanilla_grads, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_CNNVis_Vanilla_BackProp_' +name_tag+ ".png", format='png')

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_CNNVis_Vanilla_BackProp_' + visualize_method + name_tag + ".png",
                    format='png')

        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def cnn_vis_grad_cam(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                         uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library ,output_path,
                         input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, layer, affine_size, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        if type(forward_func).__name__ in self.wrapper_list:
            layer = "base." + layer

        gcam = GradCam(forward_func, target_layer=layer)
        plt.figure(figsize=(5, 5))
        idx = 1
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=GradCam,
                                            method_attribution=gcam.generate_cam,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                cam, uncertain_metrics = self.patcher_func(method_handler=gcam.generate_cam,
                                                                  params={"ip": ip, "target_class": int(i),
                                                                          "gpu_device": device, "is_3d": is_3d,
                                                                          "device": device},
                                                                  uncertainity=uncertainity if uncertainity_flag else None)
            else:
                cam = gcam.generate_cam(ip, device, is_3d, int(i))
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=cam)

            if uncertainity_flag:
                uncertain_metrics["method"] = "cnn_vis_grad_cam" + "_" +  str(i) + name_tag

            if is_3d:
                grads = cam.squeeze()
                if type(grads) is not np.ndarray:
                    grads = grads.cpu().detach().numpy().astype('float32')
                if isDepthFirst:
                    if len(grads.shape) > 3:
                        grads = np.moveaxis(grads, [0, 1], [-1, -2])
                    else:
                        grads = np.moveaxis(grads, 0, -1)

                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_CNNVis_Grad_Cam_' + str(i) + name_tag+".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_CNNVis_Grad_Cam_' + str(i)+ name_tag)
            else:

                #if len(cam.shape) == 3: cam *= 255
                if type(cam) is not np.ndarray:
                    cam = cam.cpu().detach().numpy().astype('float32')

                if isDepthFirst:
                    original = np.moveaxis(ip.squeeze().detach().cpu().numpy(), 0, -1) \
                        if len(ip.squeeze().detach().cpu().numpy().shape) == 3 \
                        else ip.squeeze().detach().cpu().numpy()
                else:
                    original = ip.detach().cpu().numpy()
                plt.subplot(1, len(target), idx)
                #plt.imshow(original, cmap='Greys')
                plt.imshow(cam, cmap='Reds', alpha=0.5)
                plt.colorbar(orientation='horizontal')
                plt.xticks([])
                plt.yticks([])
                plt.title("Class " + str(int(i)))
            idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        plt.suptitle(title)
        plt.savefig(output_path + "/" + input_file + '_CNNVis_Grad_Cam_' +name_tag+ ".png", format='png')
        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def cnn_vis_grad_times_image(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                 uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library,
                                 output_path, input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present,
                                 firstLayer, affine_size, name_tag = "_"):
        self.device = device
        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        if type(forward_func).__name__ in self.wrapper_list:
            firstLayer = 'base.' + firstLayer

        if len(firstLayer) > 0:
            modelName = [k for k, v in locals().items() if v == forward_func][0]
            firstLayer = eval(modelName + "." + firstLayer)
            VBP = VanillaBackprop(forward_func, is_3d, firstLayer)
        else:
            VBP = VanillaBackprop(forward_func, is_3d)

        fig = plt.figure(figsize=(5, 5))
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
        idx = 1
        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=VanillaBackprop,
                                            method_attribution=VBP.generate_gradients,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                vanilla_grads, _ = self.patcher_func(method_handler=VBP.generate_gradients,
                                                  params={"ip": ip, "target_class": int(i), "gpu_device": device,
                                                          "device": device},
                                                           uncertainity=uncertainity if uncertainity_flag else None)
            else:
                vanilla_grads = VBP.generate_gradients(ip, int(i), device)

            grad_times_image = vanilla_grads[0] * ip.detach().cpu().numpy()[0]
            if uncertainity_flag:
                uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=grad_times_image)
                uncertain_metrics["method"] = "cnn_vis_grad_times_image" + "_" +  str(i) + name_tag


            if is_3d:

                grads = normalize_gradient(grad_times_image.squeeze())
                if type(grads) is not np.ndarray:
                    grads = grads.cpu().detach().numpy().astype('float32')
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_CNNVis_Grad_Times_Image_' + str(i) + name_tag+ ".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_CNNVis_Grad_Times_Image_' + str(i)+name_tag)
            else:
                grad_times_image = np.transpose(grad_times_image, (1, 2, 0))

                #if len(grad_times_image.shape) == 3: grad_times_image *= 255
                #if isDepthFirst:
                #    original = np.moveaxis(inp_image.squeeze().detach().cpu().numpy(), 0, -1) \
                #        if len(inp_image.squeeze().detach().cpu().numpy().shape) == 3 \
                #        else inp_image.squeeze().detach().cpu().numpy()
                #else:
                #    original = inp_image.detach().cpu().numpy()
                #plt.subplot(1, len(target), idx)
                #plt.imshow(original, cmap='Greys')
                #plt.imshow(grad_times_image, cmap='Reds', alpha=0.5)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(grad_times_image, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_CNNVis_Grad_Times_Image' + name_tag+".png", format='png')

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_CNNVis_Grad_Times_Image_' + visualize_method + name_tag + ".png",
                    format='png')

        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def captum_layer_grad_shap(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                               uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file, is_3d,
                               isDepthFirst,
                               patcher_flag, batch_dim_present, layer, attribute_to_layer_input, affine_size, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base.' + layer

        modelName = [k for k, v in locals().items() if v == forward_func][0]

        layer_gradient_shap = LayerGradientShap(forward_func, eval(modelName + "." + layer))
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

        fig = plt.figure(figsize=(9, 6))
        idx = 1
        for i in target:
            if patcher_flag:
                attribution = self.patcher_func(method_handler=layer_gradient_shap.attribute,
                                                params={"ip": ip, "target": int(i),
                                                        "baselines": torch.zeros(1, 1, self.patch_size, self.patch_size,
                                                                                 self.patch_size).to(device),
                                                        "attribute_to_layer_input": attribute_to_layer_input,
                                                        "gpu_device": device})
            else:
                attribution = layer_gradient_shap.attribute(ip, baselines=ip * 0, target=int(i),
                                                            attribute_to_layer_input=attribute_to_layer_input)
            if is_3d:
                if attribute_to_layer_input:
                    grads = attribution[0].squeeze().cpu().detach().numpy()
                else:
                    grads = attribution.squeeze().cpu().detach().numpy()
                if isDepthFirst:
                    if len(grads.shape) > 3:
                        grads = np.moveaxis(grads, [0, 1], [-1, -2])
                    else:
                        grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Layer_grad_shap_' + str(i) +name_tag+ ".nii.gz"))
                if not len(grads.shape) > 3 :
                    create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_Layer_grad_shap_' + str(i)+ name_tag)
            else:
                if isDepthFirst:
                    attribution = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))
                else:
                    attribution = attribution.squeeze().cpu().detach().numpy()
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attribution, original_image, use_pyplot=False, sign=sign,
                                                  method=visualize_method,
                                                  show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                                  plt_fig_axis=(fig, axis))
            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Layer_grad_shap_' + visualize_method +name_tag+ ".png", format='png')

        ip.requires_grad = False

    def torchray_excitation_backprop(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                     uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file,
                                     is_3d, isDepthFirst, patcher_flag, batch_dim_present, layer, affine_size, name_tag = "_"):


        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        modelName = [k for k, v in locals().items() if v == forward_func][0]
        if type(forward_func).__name__ in self.wrapper_list:
            layer = eval(modelName + '.base.' + layer)
        else:
            layer = eval(modelName + '.' + layer) if len(layer) > 0 else layer

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if patcher_flag:  # different format please check !!!! not sure
                saliency_attr = self.patcher_func(method_handler=excitation_backprop,
                                                  params={"ip": ip, "model": forward_func, "target" : int(i),
                                                          "saliency_layer": layer, "library" : library,
                                                          "gpu_device": device })
            else:
                saliency_attr = excitation_backprop(forward_func, ip, int(i), saliency_layer=layer)
            if is_3d:
                grads = saliency_attr.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                grads = normalize_gradient(grads)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Excitation_Backprop_' + str(i) + name_tag+ ".nii.gz"))
            else:
                if isDepthFirst:
                    saliency_attr = np.transpose(saliency_attr.squeeze(axis=0).cpu().detach().numpy(), (1, 2, 0))
                else:
                    saliency_attr = saliency_attr.squeeze(axis=0).cpu().detach().numpy()


                #saliency_attr = normalize_gradient(saliency_attr)
                #plt.subplot(1, len(target), idx)
                #plt.imshow(saliency_attr, cmap=plt.cm.binary)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))
                #if show_colorbar:
                #    plt.colorbar(orientation='horizontal')

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(saliency_attr, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_Excitation_Backprop' + name_tag+ ".png", format='png')
        #ip.requires_grad = False

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Excitation_Backprop_' + visualize_method + name_tag + ".png",
                    format='png')

    def torchray_contrast_excitation_backprop(self, forward_func, target, inp_image, inp_transform_flag, transform_func, uncertainity_flag,
                                              visualize_method, sign, show_colorbar, title, device, library, output_path,
                                              input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present,
                                              layer, contrast_layer, classifier_layer, affine_size, name_tag = "_"):
        self.device = device

        self.logger.info("This method only work for classification task")
        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        modelName = [k for k, v in locals().items() if v == forward_func][0]
        if type(forward_func).__name__ in self.wrapper_list:
            layer = eval(modelName + '.base.' + layer) if layer is not None else None
            contrast_layer = eval(modelName + '.base.' + contrast_layer) if contrast_layer is not None else None
            classifier_layer = eval(modelName + '.base.' + classifier_layer) if classifier_layer is not None else None

        else:
            layer = eval(modelName + '.' + layer) if layer is not None else None
            contrast_layer = eval(modelName + '.' + contrast_layer) if contrast_layer is not None else None
            classifier_layer = eval(modelName + '.' + classifier_layer) if classifier_layer is not None else None

        for i in target:
            if patcher_flag:
                saliency_attr = self.patcher_func(method_handler=contrastive_excitation_backprop,
                                                  params={"ip": ip, "target": int(i), "forward_func": forward_func,
                                                          "saliency_layer": layer, "contrast_layer": contrast_layer,
                                                          "classifier_layer": classifier_layer, "gpu_device": device})
            else:
                saliency_attr = contrastive_excitation_backprop(forward_func, ip, int(i), saliency_layer=layer,
                                                                contrast_layer=contrast_layer,
                                                                classifier_layer=classifier_layer)
            if is_3d:
                grads = saliency_attr.squeeze().cpu().detach().numpy()
                grads = normalize_gradient(grads)
                if isDepthFirst:
                    grads = np.moveaxis(grads, 0,-1)

                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path,
                                          input_file + '_Contrastive_Excitation_Backprop_' + str(i) + name_tag+ ".nii.gz"))
            else:
                saliency_attr = np.transpose(saliency_attr.squeeze(axis=0).cpu().detach().numpy(), (1, 2, 0))

                #saliency_attr = normalize_gradient(saliency_attr)
                #plt.subplot(1, len(target), idx)
                #plt.imshow(saliency_attr)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))
                #if show_colorbar:
                #    plt.colorbar(orientation='horizontal')

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(saliency_attr, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_Contrastive_Excitation_Backprop' +name_tag+ ".png", format='png')

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Contrastive_Excitation_Backprop_' + visualize_method + name_tag + ".png",
                    format='png')

        ip.requires_grad = False

    def captum_gradient_shap(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                             uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file,
                             is_3d, isDepthFirst, patcher_flag, batch_dim_present, n_samples, stdiv=0.0, affine_size=4, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        grad_shap = GradientShap(forward_func)

        if patcher_flag:
            baseline1 = torch.ones((1, 1,self.patch_size, self.patch_size, self.patch_size)).to(device)
            rand_img_dist = torch.cat([baseline1*0, baseline1])
        else:
            rand_img_dist = torch.cat([ip * 0, ip * 1])

        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=GradCam,
                                            method_attribution=grad_shap.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                attribution, uncertain_metrics = self.patcher_func(method_handler=grad_shap.attribute, params={"ip": ip, "target": int(i),
                                                                                            "baselines": rand_img_dist,
                                                                                            "n_samples": n_samples,
                                                                                            "stdevs": stdiv,
                                                                                            "gpu_device": device},
                                                           uncertainity=uncertainity if uncertainity_flag else None)
            else:
                attribution = grad_shap.attribute(ip, baselines=rand_img_dist, n_samples=n_samples, stdevs=stdiv,
                                                  target=int(i))
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=attribution, baselines=rand_img_dist, n_samples=n_samples, stdevs=stdiv)

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_gradient_shap" + "_" +  str(i) + name_tag


            if is_3d:
                grads = attribution.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Gradient_shap_' + str(i) + name_tag+".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_Gradient_shap_' + str(i)+ name_tag)
            else:
                if isDepthFirst:
                    attribution = np.moveaxis(attribution.squeeze().cpu().detach().numpy(), 0, -1)
                else:
                    attribution = attribution.squeeze().cpu().detach().numpy()

                #plt.subplot(1, len(target), idx)
                #plt.imshow(original_image, cmap='Greys')
                #plt.imshow(attribution, cmap='Reds', alpha=0.5)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attribution, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Gradient_shap_' + visualize_method +name_tag+ ".png", format='png')

        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def captum_internal_influence(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                  uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file,
                                  is_3d, isDepthFirst, patcher_flag, batch_dim_present, layer, device_ids, affine_size, name_tag = "_"):

        if len(device_ids) > 1 and (device.split(':')[-1] != str(device_ids[0])):
            raise Exception(
                "For multi device execution your device id must match the fist device number of the device id list")
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        modelName = [k for k, v in locals().items() if v == forward_func][0]
        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base.' + layer

        if (len(device_ids) > 0):
            forward_func = nn.DataParallel(forward_func, device_ids)
            int_inf = InternalInfluence(forward_func, eval(modelName + ".module." + layer), device_ids)
        else:
            int_inf = InternalInfluence(forward_func, eval(modelName + "." + layer))
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=GradCam,
                                            method_attribution=int_inf.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                attribution, uncertain_metrics = self.patcher_func(method_handler=int_inf.attribute,
                                                params={"ip": ip, "target": int(i), "gpu_device": device},
                                                           uncertainity=uncertainity if uncertainity_flag else None)
            else:
                attribution = int_inf.attribute(ip, target=int(i))
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=attribution)

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_internal_influence"+ "_" +  str(i) + name_tag

            if is_3d:
                grads = attribution.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Internal_influence_' + str(i) +name_tag+ ".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_Internal_influence_' + str(i)+ name_tag)
            else:
                if isDepthFirst:
                    attribution = np.moveaxis(attribution.squeeze().cpu().detach().numpy(), 0, -1)
                else:
                    attribution = attribution.squeeze().cpu().detach().numpy()

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

                #plt.subplot(1, len(target), idx)
                #plt.imshow(original_image, cmap='Greys')
                #plt.imshow(attribution, cmap='Reds', alpha=0.5)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attribution, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1

        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Internal_influence_' + visualize_method +name_tag+ ".png", format='png')

        ip.requires_grad = False
        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def captum_input_x_gradient(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file,
                                is_3d, isDepthFirst, patcher_flag, batch_dim_present, affine_size,name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        inp_x_grad = InputXGradient(forward_func)
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=GradCam,
                                            method_attribution=inp_x_grad.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                attribution, uncertain_metrics = self.patcher_func(method_handler=inp_x_grad.attribute,
                                                params={"ip": ip, "target": int(i), "gpu_device": device},
                                                                   uncertainity=uncertainity if uncertainity_flag else None)
            else:
                attribution = inp_x_grad.attribute(ip, target=int(i))
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=attribution)

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_input_x_gradient" + "_" +  str(i) + name_tag

            if is_3d:
                grads = attribution.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Input_x_grad_' + str(i) +name_tag+ ".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_Input_x_grad_' + str(i)+name_tag)
            else:
                if isDepthFirst:
                    attribution = np.moveaxis(attribution.squeeze().cpu().detach().numpy(), 0, -1)
                else:
                    attribution = attribution.squeeze().cpu().detach().numpy()

                #plt.subplot(1, len(target), idx)
                #plt.imshow(original_image, cmap='Greys')
                #plt.imshow(attribution, cmap='Reds', alpha=0.5)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attribution, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_Input_x_grad_' + visualize_method +name_tag+ ".png", format='png')
        ip.requires_grad = False

        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def captum_deep_lift_shap(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                              uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file,
                              is_3d, isDepthFirst, patcher_flag, batch_dim_present, affine_size, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        dLiftshap = DeepLiftShap(forward_func)
        if patcher_flag:
            baseline1 = torch.ones((1, 1,self.patch_size, self.patch_size, self.patch_size)).to(device)
            rand_img_dist = torch.cat([baseline1*0, baseline1* 1e-5])
        else:
            rand_img_dist = torch.cat([ip * 0, ip * 1e-5])
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=GradCam,
                                            method_attribution=dLiftshap.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                attribution, uncertain_metrics = self.patcher_func(method_handler=dLiftshap.attribute,
                                                params={"ip": ip, "target": int(i), "baselines": rand_img_dist,
                                                        "gpu_device": device},
                                                                   uncertainity=uncertainity if uncertainity_flag else None)
            else:
                attribution = dLiftshap.attribute(ip, baselines=rand_img_dist, target=int(i))
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=attribution)

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_deep_lift_shap"+ "_" +  str(i) + name_tag

            if is_3d:
                grads = attribution.squeeze().cpu().detach().numpy()
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Deep_lift_shap_' + str(i) +name_tag+ ".nii.gz"))
            else:
                if isDepthFirst:
                    attribution = np.moveaxis(attribution.squeeze().cpu().detach().numpy(), 0, -1)
                else:
                    attribution = attribution.squeeze().cpu().detach().numpy()

                #plt.subplot(1, len(target), idx)
                #plt.imshow(original_image, cmap='Greys')
                #plt.imshow(attribution, cmap='Reds', alpha=0.5)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attribution, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_deep_lift_shap_' + visualize_method +name_tag+ ".png", format='png')
        ip.requires_grad = False
        if uncertainity_flag:
            metrics = self.uncertain_metrics
            self.uncertain_metrics = []
            return metrics
        else:
            return None

    def lucent_render_vis(self, forward_func, target=None, inp_image=None, inp_transform_flag=None, transform_func=None, uncertainity_flag=False,
                          visualize_method=None, sign=None, show_colorbar=None, title=None, device=None, library="", output_path='',
                          input_file='', is_3d=False, isDepthFirst=False, patcher_flag=None, batch_dim_present=None,
                          layer=None, affine_size=4, shape=None, name_tag = "_"):
        if is_3d:
            if shape is None:
                self.logger.error("Desired shape cannot be blank for 3D models")
                return
        forward_func.to(device).eval()

        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base_' + layer
        if is_3d:
            _ = render.render_vis(forward_func, layer, save_image=True, progress=False, show_image=False,
                                  image_name=output_path + "/" + input_file + "_" + str(
                                      layer.split(":")[0]) + '_lucent_vis.nii.gz', is_3d=is_3d,
                                  isDepthFirst=isDepthFirst, shape=shape, affine_size=affine_size)
        else:
            _ = render.render_vis(forward_func, layer, save_image=True, progress=False, show_image=False,
                                  image_name=output_path + "/" + input_file + '_lucent_vis.png', is_3d=is_3d,
                                  isDepthFirst=isDepthFirst, shape=shape)

    def captum_LayerGradientXActivation(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                        uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path, input_file,
                                        is_3d, isDepthFirst,
                                        patcher_flag, batch_dim_present, layer, affine_size, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        modelName = [k for k, v in locals().items() if v == forward_func][0]

        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base.' + layer

        lgc = LayerGradientXActivation(forward_func, eval(modelName + "." + layer))
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if patcher_flag:
                attribution = self.patcher_func(method_handler=lgc.attribute,
                                                params={"ip": ip, "target": int(i), "gpu_device": device})
            else:
                attribution = lgc.attribute(ip, target=int(i))
            if is_3d:
                grads = attribution.squeeze().cpu().detach().numpy()
                if isDepthFirst:
                    if len(grads.shape) > 3:
                        grads = np.moveaxis(grads, [0,1], [-1,-2])
                    else:
                        grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_layer_grad_x_activation_' + str(i) +name_tag+ ".nii.gz"))
            else:
                if isDepthFirst:
                    attribution = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))
                else:
                    attribution = attribution.squeeze().cpu().detach().numpy()
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attribution, original_image, use_pyplot=False, sign=sign,
                                                  method=visualize_method,
                                                  show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                                  plt_fig_axis=(fig, axis))
            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_layer_grad_x_activation_' + visualize_method+name_tag + ".png",
                    format='png')

        ip.requires_grad = False

    def captum_LayerDeepLift(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                             uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path,
                             input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, layer,
                             attribute_to_layer_input, affine_size, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        modelName = [k for k, v in locals().items() if v == forward_func][0]

        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base.' + layer

        ldl = LayerDeepLift(forward_func, eval(modelName + "." + layer))
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if patcher_flag:
                attribution = self.patcher_func(method_handler=ldl.attribute, params={"ip": ip, "target": int(i),
                                                                                      "attribute_to_layer_input": attribute_to_layer_input,
                                                                                      "gpu_device": device})
            else:
                attribution = ldl.attribute(ip, target=int(i), attribute_to_layer_input=attribute_to_layer_input)
            if type(attribution) is tuple: attribution = attribution[0]
            if is_3d:
                grads = attribution.squeeze().cpu().detach().numpy()
                if isDepthFirst:
                    if len(grads.shape) >3:
                        grads = np.moveaxis(grads, [0,1], [-1, -2])
                    else:
                        grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_layer_deep_lift_' + str(i) + name_tag+".nii.gz"))

                #create_max_projection(grads,
                #                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                #                      else inp_image.detach().squeeze().to('cpu').numpy(),
                #                      output_path, input_file + 'ldl_' + str(i) + name_tag)

            else:
                if isDepthFirst:
                    attribution = np.moveaxis(attribution.squeeze().cpu().detach().numpy(), 0, -1)
                else:
                    attribution = attribution.squeeze().cpu().detach().numpy()

                #plt.subplot(1, len(target), idx)
                #plt.imshow(original_image, cmap='Greys')
                #plt.imshow(attribution, cmap='Reds', alpha=0.5)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attribution, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_layer_deep_lift_' + visualize_method + name_tag+ ".png",
                    format='png')

        ip.requires_grad = False

    def captum_LayerGradCam(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                            uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path,
                            input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, layer, affine_size, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        modelName = [k for k, v in locals().items() if v == forward_func][0]
        if type(forward_func).__name__ in self.wrapper_list:
            layer = 'base.' + layer

        lgcam = LayerGradCam(forward_func, eval(modelName + "." + layer))
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if patcher_flag:
                attribution, _ = self.patcher_func(method_handler=lgcam.attribute,
                                                params={"ip": ip, "target": int(i), "gpu_device": device})
            else:
                attribution = lgcam.attribute(ip, target=int(i))
            if is_3d:
                grads = normalize_gradient(attribution.squeeze().cpu().detach().numpy())
                if isDepthFirst:
                    if len(grads.shape) > 3:
                        grads = np.moveaxis(grads, [0, 1], [-1, -2])
                    else:
                        grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_layer_grad_cam_' + str(i) + name_tag+".nii.gz"))
                original_image = np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst else inp_image.detach().squeeze().to('cpu').numpy()
                if original_image.shape == grads.shape:
                    create_max_projection(grads,original_image,
                                          output_path, input_file + '_layer_grad_cam_captum_' + str(i) + name_tag)

            else:
                if isDepthFirst:
                    attribution = np.moveaxis(attribution.squeeze(0).cpu().detach().numpy(), 0, -1)
                else:
                    attribution = attribution.squeeze(0).cpu().detach().numpy()

                #plt.subplot(1, len(target), idx)
                #plt.imshow(original_image, cmap='Greys')
                #plt.imshow(attribution, cmap='Reds', alpha=0.5)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))

                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(attribution, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_layer_grad_cam_' + visualize_method +name_tag+ ".png",
                    format='png')

        ip.requires_grad = False

    def torchray_Rise(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                      uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path,
                      input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, affine_size=4, name_tag = "_"):
        self.device = device

        if is_3d:
            self.logger.info("This method can't be run for 3d Model.")
            self.logger.info("The underlying library doesn't support 3d spatial shape.")
            self.logger.info("Code internal assertion stops the methods to be extended to 3d models.")
            return

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        plt.figure(figsize=(9, 6))
        idx = 1
        if patcher_flag:
            saliency_attr_full = self.patcher_func(method_handler=rise,
                                                   params={"forward_func": forward_func, "target": ip,
                                                           "gpu_device": device})
        else:
            saliency_attr_full = rise(forward_func, ip)

        for i in target:
            if is_3d:
                saliency_attr = saliency_attr_full[:, i].cpu().detach().numpy()
            else:
                saliency_attr = np.transpose(saliency_attr_full[:, i].cpu().detach().numpy(), (1, 2, 0))

            saliency_attr = normalize_gradient(saliency_attr)
            if is_3d:
                grads = saliency_attr.squeeze().cpu().detach().numpy()
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Rise_' + str(i) +name_tag+ ".nii.gz"))
            else:
                plt.subplot(1, len(target), idx)
                plt.imshow(saliency_attr, cmap=plt.cm.binary)
                plt.xticks([])
                plt.yticks([])
                plt.title("Class " + str(int(i)))
                if show_colorbar:
                    plt.colorbar(orientation='horizontal')
            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        plt.suptitle(title)
        plt.savefig(output_path + "/" + input_file + '_Rise' +name_tag+ ".png", format='png')
        ip.requires_grad = False

    def torchray_Deconv(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                        uncertainity_flag, visualize_method, sign, show_colorbar, title, device,library, output_path,
                        input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, affine_size=4, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if patcher_flag:
                saliency_attr = self.patcher_func(method_handler=deconvnet,
                                                  params={"ip": ip, "target": int(i), "gpu_device": device})
            else:
                saliency_attr = deconvnet(forward_func, ip, int(i))
            if is_3d:
                grads = saliency_attr.squeeze().cpu().detach().numpy()
                grads = normalize_gradient(grads)
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_Torchray_deconvNet_' + str(i) +name_tag+ ".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_Torchray_deconvNet_' + str(i)+ name_tag)
            else:
                if isDepthFirst:
                    saliency_attr = np.transpose(saliency_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                else:
                    saliency_attr = saliency_attr.squeeze(0).cpu().detach().numpy()

                #saliency_attr = normalize_gradient(saliency_attr)
                #plt.subplot(1, len(target), idx)
                #plt.imshow(saliency_attr, cmap=plt.cm.binary)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))
                #if show_colorbar:
                #    plt.colorbar(orientation='horizontal')

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(saliency_attr, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_torchray_deconvNet' +name_tag+ ".png", format='png')

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_torchray_deconvNet_' + visualize_method + name_tag + ".png",
                    format='png')

        ip.requires_grad = False

    def torchray_Gradcam(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                         uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path,
                         input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, layer, affine_size=4, name_tag = "_"):

        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        modelName = [k for k, v in locals().items() if v == forward_func][0]
        if type(forward_func).__name__ in self.wrapper_list:
            layer = eval(modelName + ".base." + layer)
        else:
            layer = eval(modelName + '.' + layer)

        for i in target:
            if patcher_flag:
                saliency_attr = self.patcher_func(method_handler=grad_cam,
                                                  params={"ip": ip, "target": int(i), "forward_func": forward_func,
                                                          "saliency_layer": layer, "gpu_device": device})
            else:
                saliency_attr = grad_cam(forward_func, ip, int(i), saliency_layer=layer)
            if is_3d:
                grads = saliency_attr.squeeze().cpu().detach().numpy()
                grads = normalize_gradient(grads)
                # print(grads.shape)
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_torchray_gradcam_' + str(i) +name_tag+ ".nii.gz"))
            else:
                if isDepthFirst:
                    saliency_attr = np.transpose(saliency_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                else:
                    saliency_attr = saliency_attr.squeeze(0).cpu().detach().numpy()
                # saliency_attr = normalize_gradient(saliency_attr)
                #plt.subplot(1, len(target), idx)
                #plt.imshow(saliency_attr, cmap='Greys')
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))
                #if show_colorbar:
                #    plt.colorbar(orientation='horizontal')

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(saliency_attr, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_torchray_gradcam' + name_tag+".png", format='png')

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_torchray_gradcam_' + visualize_method + name_tag + ".png",
                    format='png')

        ip.requires_grad = False

    def torchray_Gradient(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                          uncertainity_flag, visualize_method, sign, show_colorbar, title, device, library, output_path,
                          input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, affine_size=4, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if patcher_flag:
                saliency_attr = self.patcher_func(method_handler=gradient,
                                                  params={"forward_func": forward_func, "ip": ip, "target": int(i),
                                                          "gpu_device": device})
            else:
                saliency_attr = gradient(forward_func, ip, int(i))
            if is_3d:
                grads = saliency_attr.squeeze().cpu().detach().numpy()
                grads = normalize_gradient(grads)
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_torchray_gradient_' + str(i) +name_tag+ ".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_torchray_gradient_' + str(i)+name_tag)
            else:
                if isDepthFirst:
                    saliency_attr = np.transpose(saliency_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                else:
                    saliency_attr = saliency_attr.squeeze(0).cpu().detach().numpy()

                #saliency_attr = normalize_gradient(saliency_attr)
                #plt.subplot(1, len(target), idx)
                #plt.imshow(saliency_attr, cmap='Greys')
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))
                #if show_colorbar:
                #    plt.colorbar(orientation='horizontal')

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(saliency_attr, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return

        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_torchray_gradient' +name_tag+ ".png", format='png')

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_torchray_gradient_' + visualize_method + name_tag + ".png",
                    format='png')

        ip.requires_grad = False

    def torchray_Guidedbackprop(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                uncertainity_flag,visualize_method, sign, show_colorbar, title, device,library, output_path,
                                input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, affine_size=4,name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:
            if patcher_flag:
                saliency_attr = self.patcher_func(method_handler=guided_backprop,
                                                  params={"ip": ip, "target": int(i), "forward_func": forward_func,
                                                          "gpu_device": device})
            else:
                saliency_attr = guided_backprop(forward_func, ip, int(i))
            if is_3d:
                grads = saliency_attr.squeeze().cpu().detach().numpy()
                grads = normalize_gradient(grads)
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_torchray_guided_backprop_' + str(i) +name_tag+ ".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_torchray_guided_backprop_' + str(i)+name_tag)
            else:
                if isDepthFirst:
                    saliency_attr = np.transpose(saliency_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                else:
                    saliency_attr = saliency_attr.squeeze(0).cpu().detach().numpy()

                #saliency_attr = normalize_gradient(saliency_attr)
                #plt.subplot(1, len(target), idx)
                #plt.imshow(saliency_attr, cmap='binary')
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))
                #if show_colorbar:
                #    plt.colorbar(orientation='horizontal')

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(saliency_attr, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_torchray_guided_backprop' +name_tag+ ".png", format='png')

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_torchray_guided_backprop_' + visualize_method + name_tag + ".png",
                    format='png')

        ip.requires_grad = False

    def torchray_Linearapprox(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                              uncertainity_flag,visualize_method, sign, show_colorbar, title, device, library,
                              output_path, input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, layer,
                              affine_size=4, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        modelName = [k for k, v in locals().items() if v == forward_func][0]
        if type(forward_func).__name__ in self.wrapper_list:
            layer = eval(modelName + ".base." + layer)
        else:
            layer = eval(modelName + "." + layer)

        for i in target:
            if patcher_flag:
                saliency_attr = self.patcher_func(method_handler=linear_approx,
                                                  params={"ip": ip, "target": int(i), "forward_func": forward_func,
                                                          "saliency_layer": layer, "gpu_device": device})
            else:
                saliency_attr = linear_approx(forward_func, ip, int(i), saliency_layer=layer)
            if is_3d:
                grads = saliency_attr.squeeze().cpu().detach().numpy()
                grads = normalize_gradient(grads)
                if isDepthFirst: grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path, input_file + '_torchray_linear_approx_' + str(i)+name_tag + ".nii.gz"))
            else:
                if isDepthFirst:
                    saliency_attr = np.transpose(saliency_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
                else:
                    saliency_attr = saliency_attr.squeeze(0).cpu().detach().numpy()

                #saliency_attr = normalize_gradient(saliency_attr)
                #plt.subplot(1, len(target), idx)
                #plt.imshow(saliency_attr, cmap=plt.cm.binary)
                #plt.xticks([])
                #plt.yticks([])
                #plt.title("Class " + str(int(i)))
                #if show_colorbar:
                #    plt.colorbar(orientation='horizontal')

                original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))
                axis = fig.add_subplot(1, len(target), idx)
                fig, _ = viz.visualize_image_attr(saliency_attr, original_image, use_pyplot=False, method=visualize_method,
                                                  sign=sign,
                                                  show_colorbar=show_colorbar, title='Class ' + str(i),
                                                  plt_fig_axis=(fig, axis))

            idx += 1
        if is_3d:
            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        #plt.suptitle(title)
        #plt.savefig(output_path + "/" + input_file + '_torchray_linear_approx' +name_tag+ ".png", format='png')

        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_torchray_linear_approx_' + visualize_method + name_tag + ".png",
                    format='png')

        ip.requires_grad = False

    def cnn_vis_layer_activation_guided_backprop(self, forward_func, target, inp_image, inp_transform_flag,
                                                 transform_func, uncertainity_flag,
                                                 visualize_method, sign, show_colorbar, title, device, library, output_path,
                                                 input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present,
                                                 layer, filter_pos, firstLayer, affine_size=4, name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()
        modelName = [k for k, v in locals().items() if v == forward_func][0]
        if type(forward_func).__name__ in self.wrapper_list:
            layer = "base." + layer
            firstLayer = "base." + firstLayer

        if len(firstLayer) > 0:
            firstLayer = eval(modelName + "." + firstLayer)
            gbp = Cnn_Vis_LaGBP(forward_func, is_3d, firstLayer, eval(modelName + "." + layer))
        else:
            gbp = Cnn_Vis_LaGBP(forward_func, is_3d, None, eval(modelName + "." + layer))
        plt.figure(figsize=(5, 5))
        idx = 1
        for i in target:
            if patcher_flag:
                grads,_ = self.patcher_func(method_handler=gbp.generate_gradients,
                                          params={"ip": ip, "target_class": int(i), "cnn_layer": layer, "filter_pos": filter_pos,
                                                  "gpu_device": device})
            else:
                grads = gbp.generate_gradients(ip, int(i), layer, filter_pos)
            if is_3d:


                grads = grads.squeeze()
                if type(grads) is not np.ndarray:
                    grads = grads.numpy().astype('float32')

                #grads = normalize_gradient(grads)

                if isDepthFirst:
                    if len(grads.shape) > 3:
                        grads = np.moveaxis(grads, [0, 1], [-1, -2])
                    else:
                        grads = np.moveaxis(grads, 0, -1)
                img = nb.Nifti1Image(grads, np.eye(affine_size))
                nb.save(img, os.path.join(output_path,
                                          input_file + '_CNNVis_LayerActivation_Guided_BackProp_' + str(i) + name_tag+".nii.gz"))
                create_max_projection(grads,
                                      np.moveaxis(inp_image.detach().squeeze().to('cpu').numpy(), 0, -1) if isDepthFirst
                                      else inp_image.detach().squeeze().to('cpu').numpy(),
                                      output_path, input_file + '_CNNVis_LayerActivation_Guided_BackProp_' + str(i)+name_tag)
            else:
                if isDepthFirst and len(grads.shape) > 2:
                    grads = normalize_gradient(np.transpose(grads, (1, 2, 0)))
                else:
                    grads = normalize_gradient(grads)

                #if len(grads.shape) ==3 : grads *= 255

                if isDepthFirst:
                    original = np.moveaxis(ip.squeeze().detach().cpu().numpy(), 0, -1) \
                        if len(ip.squeeze().detach().cpu().numpy().shape) == 3 \
                        else ip.squeeze().detach().cpu().numpy()
                else:
                    original = ip.detach().cpu().numpy()
                plt.subplot(1, len(target), idx)
                #plt.imshow(original, cmap='Greys')
                plt.imshow(grads)
                plt.xticks([])
                plt.yticks([])
                if show_colorbar:
                    plt.colorbar()
                plt.title("Class " + str(int(i)))

            idx += 1
        if is_3d:

            if uncertainity_flag:
                metrics = self.uncertain_metrics
                self.uncertain_metrics = []
                return metrics
            return
        plt.suptitle(title)
        plt.savefig(output_path + "/" + input_file + '_CNNVis_LayerActivation_Guided_BackProp' +name_tag+ ".png", format='png')



    def cnn_vis_layer_visualization(self, forward_func, target, inp_image, inp_transform_flag,
                                    transform_func, uncertainity_flag, visualize_method, sign, show_colorbar, title,
                                    device, library, output_path, input_file, is_3d, isDepthFirst, patcher_flag,
                                    batch_dim_present, layer, filter_pos, affine_size=4,name_tag = "_"):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        if type(forward_func).__name__ in self.wrapper_list:
            layer = "base." + layer

        if patcher_flag:
            layer_vis = self.patcher_func(method_handler=CNNLayerVisualization,
                                          params={"ip": ip,"forward_func": forward_func, "layer": layer,
                                                  "filter_pos": filter_pos,
                                                  "gpu_device": device})
        else:
            layer_vis = CNNLayerVisualization(forward_func, layer, filter_pos)
        if is_3d:
            layer_vis.visualise_layer_with_hooks(device, output_path + "/", True, tuple(ip.size()), affine_size, isDepthFirst)
            #layer_vis.visualise_layer_without_hooks(device, output_path + "/", True, tuple(ip.size()), affine_size)
        else:
            layer_vis.visualise_layer_with_hooks(device, output_path + "/", isDepthFirst= isDepthFirst)
            layer_vis.visualise_layer_without_hooks(device, output_path + "/", isDepthFirst=isDepthFirst)

    def cnn_vis_DeepDream(self, forward_func, target, inp_image, inp_transform_flag,
                          transform_func, uncertainity_flag,  visualize_method, sign, show_colorbar,
                          title, device, library, output_path, input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present,
                          layer, filter_pos, init_color_low, init_color_high, shape, affine_size=4, update_count = 250, method= None):
        self.device = device

        if inp_transform_flag:
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        if type(forward_func).__name__ in self.wrapper_list:
            layer = "base." + layer

        dd = DeepDream(forward_func, layer, filter_pos, output_path + "/", init_color_low, init_color_high,
                       tuple(shape), method)
        if is_3d:
            dd.dream(device, affine_size, is_3d=True, update_count = update_count, is_depth_first= isDepthFirst)
        else:
            dd.dream(device, update_count = update_count, is_depth_first= isDepthFirst)

    def captum_ShapleyValueSampling(self, forward_func, target, inp_image, inp_transform_flag, transform_func,
                                  uncertainity_flag,  visualize_method, sign, show_colorbar, title, device, library, output_path,
                                    input_file, is_3d, isDepthFirst, patcher_flag, batch_dim_present, n_samples, affine_size, name_tag = "_"):

        self.device = device

        if inp_transform_flag:
            # print(inp_image.shape)
            ip = transform_func(inp_image).to(device)
        else:
            ip = inp_image.to(device)

        if not batch_dim_present:
            ip = ip.unsqueeze(0)

        ip.requires_grad = True
        forward_func.to(device).eval()

        modelName = [k for k, v in locals().items() if v == forward_func][0]

        svs = ShapleyValueSampling(forward_func)
        original_image = np.transpose(ip.squeeze().cpu().detach().numpy(), (1, 2, 0))

        fig = plt.figure(figsize=(9, 6))
        idx = 1

        for i in target:

            if uncertainity_flag:
                uncertainity = Uncertainity(forward_func=forward_func, method=GradCam,
                                            method_attribution=svs.attribute,
                                            patcher_func=self.patcher_func, device=device, library=library,
                                            batch_dim_present=batch_dim_present, output_path=output_path,
                                            input_file=input_file,
                                            is_3d=is_3d, isDepthFirst=isDepthFirst, patcher_flag=patcher_flag, abs=abs,
                                            affine_size=affine_size,
                                            visualize_method=visualize_method, sign=sign, show_colorbar=show_colorbar)
            if patcher_flag:
                attribution, uncertain_metrics = self.patcher_func(method_handler=svs.attribute,
                                                params={"ip": ip, "target": int(i), "gpu_device": device, "n_samples":n_samples},
                                                           uncertainity=uncertainity if uncertainity_flag else None)
            else:
                attribution = svs.attribute(ip, target=int(i), n_samples=n_samples)
                if uncertainity_flag:
                    uncertain_metrics = uncertainity.get_metrics(ip=ip, target=int(i), grads=attribution)

            if uncertainity_flag:
                uncertain_metrics["method"] = "captum_ShapleyValueSampling" + "_" +  str(i) + name_tag

            attribution = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))
            axis = fig.add_subplot(1, len(target), idx)
            fig, _ = viz.visualize_image_attr(attribution, original_image, use_pyplot=False, sign=sign,
                                              method=visualize_method,
                                              show_colorbar=show_colorbar, title="Class " + str(int(i)),
                                              plt_fig_axis=(fig, axis))
            idx += 1
        fig.suptitle(title)
        fig.savefig(output_path + "/" + input_file + '_shapleyValue_Sampling_' + visualize_method + name_tag+".png",
                    format='png')

        ip.requires_grad = False
