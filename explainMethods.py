#########################################################################
## Project: Explain-ability and Interpret-ability for segmentation models
## Purpose: Ensemble python file for all functions present in thrid party libraries
## 			Methods extends functions from third party libraries
## Author: Arnab Das
#########################################################################

import torch
import torchvision.transforms as transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import numpy as np
import logging
import os

class Explainability:
  def __init__(self, model):
    self.model = model

  def inpTransformSetupLime(self,resize, centre_crop, mean_vec=None, std_vec=None):
    return transforms.Compose(
        [transforms.Resize(resize),
         transforms.CenterCrop(centre_crop)])

  def inpTransformSetup(self, resize, centre_crop, mean_vec, std_vec):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(centre_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vec, std=std_vec),
    ])


  def lime_segmentation(self,forward_func, inp_image, inp_transform_flag,transform_func, device,
                        output_path, input_file, tran_func, is_3d, target, batch_dim_present,
                        top_labels, num_samples, num_features, depth=None):

    self.device = device
    indx = 0
    target = target if type(target) == int else int(target[0])
    if inp_transform_flag:
      ip = transform_func(inp_image).to(device)
    else:
      ip = inp_image.to(device)

    if not batch_dim_present:
        ip = ip.unsqueeze(0)

    def batch_func(imgs):
      forward_func.to(device)
      imgs = torch.from_numpy(imgs).transpose(-1,1).float().to(device)
      op = forward_func(imgs).detach().cpu().numpy()
      return op

    def batch_func_3d(imgs):
      forward_func.to(device)
      outputArray = []
      for i in range(imgs.shape[0]):
          ip[0,0,indx]= torch.from_numpy(imgs[i,:,:,0]).float().to(device)
          outputArray.append(forward_func(ip).detach().cpu().numpy().transpose())
      return np.array(outputArray).squeeze(-1)


    explainer = lime_image.LimeImageExplainer()
    if not is_3d:
      if type(inp_image) == torch.Tensor:
          input = np.array(inp_image.squeeze().type(torch.DoubleTensor).cpu())
          if input.shape[0] == 3:
              input = np.moveaxis(input, 0, -1)
      else:
          input = np.array(tran_func(inp_image))
      explanation = explainer.explain_instance(input,
                                             batch_func,
                                             top_labels=top_labels,
                                             hide_color=0,
                                             num_samples=num_samples,)
      temp, mask = explanation.get_image_and_mask(target, positive_only=True,
                                                num_features=num_features, hide_rest=True)
      img_boundry1 = mark_boundaries(temp / 255.0, mask)
      save_im = Image.fromarray((img_boundry1 * 255).astype(np.uint8))
      save_im.save(
          f"{output_path}/{input_file}_lime__towards_prediction_class_{str(target)}.png",
          format='png',
      )

      temp, mask = explanation.get_image_and_mask(target, positive_only=False,
                                                num_features=num_features, hide_rest=False)
      img_boundry2 = mark_boundaries(temp / 255.0, mask)
      save_im = Image.fromarray((img_boundry2 * 255).astype(np.uint8))
      save_im.save(
          f"{output_path}/{input_file}_lime__against_prediction_class_{str(target)}.png",
          format='png',
      )
    else:
      #print(target, type(target))
      if not batch_dim_present:
          inp_image = inp_image.unsqueeze(0)
      if not os.path.exists(f'{output_path}/Lime_{input_file}'):
        os.mkdir(f'{output_path}/Lime_{input_file}')
      while indx< depth:
        explanation = explainer.explain_instance(np.array(inp_image[0,0,indx].type(torch.DoubleTensor)),
                                             batch_func_3d,
                                             top_labels=top_labels,
                                             hide_color=0,
                                             num_samples=num_samples)
        temp, mask = explanation.get_image_and_mask(target, positive_only=False,
                                                    num_features=num_features, hide_rest=False)
        img_boundry1 = mark_boundaries(temp / 255.0, mask)
        save_im = Image.fromarray((img_boundry1 * 255).astype(np.uint8))
        save_im.save(
            f'{output_path}/Lime_{input_file}/{indx}_towards_prediction_class_{str(target)}.png',
            format='png',
        )

        indx +=1




