#########################################################################
## Project: Explain-ability and Interpret-ability for segmentation models
## Purpose: Ensemble python file for all functions present in thrid party libraries
## 			Methods extends functions from third party libraries
## Author: Arnab Das
#########################################################################

import os
import sys

from scipy.ndimage import zoom
from torchvision import models
from methods import Interpretability
from explainMethods import Explainability
from model_copy import Net, Bottleneck
from makeCustomModel import NetCustom
from PIL import Image
import json
import torch
import logging
import warnings
from tqdm import tqdm
from torch import nn
import numpy as np
import os.path as osp
import pickle
from functools import partial
from NiftyFilerLoader import R2Plus1dStem4MRI, load_checkpoint
import nibabel as nb
import torchvision.transforms as transforms
from SegmentModel.VesselSeg_UNet3d import U_Net


with open('cfg_Misc_Class_2d.json') as f:
    config = json.load(f)
model = None
inp = None
inp_path = None

if config["default"] and not config["is_3d"]:
    if config['model_nature'] == 'Segmentation':
        inp_path = 'testImage.jpg'
        inp = Image.open(inp_path)
        resModel = models.segmentation.fcn_resnet101(pretrained=True)
        model = Net(resModel)
    if config['model_nature'] == 'Classification':
        model = models.resnet18(pretrained=True)
        inp_path = 'imagenetImage.jpg'
        inp = Image.open(inp_path)
        model.eval()
elif config["default"]:
    if config['model_nature'] == 'Classification':
        inp_path = 'IXI012-HH-1211-T1_brain_resampled.nii.gz'
        inp = nb.load(inp_path)
        inp = inp.get_fdata()
        inp  =zoom(inp, (0.4, 0.4, 0.5))
        inp = torch.from_numpy(inp)
        inp = torch.unsqueeze(inp, 0).float()
        model = models.video.r3d_18(pretrained=True)
        model.stem = R2Plus1dStem4MRI()
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 3)
        )
        # model_3d.to('cuda:0')

        ckpt = load_checkpoint('resnet3D_0.001')
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
    if config['model_nature'] == 'Segmentation':
        inp_path = 'SegmentModel/data/IXI425-IOP-0988-MRA.nii.gz'
        inp = nb.load(inp_path)
        inp = inp.get_fdata()
        inp = zoom(inp, (0.125, 0.125, 0.35))
        inp = (inp - np.min(inp)) / (np.max(inp) - np.min(inp))
        inp = np.moveaxis(inp, -1, 0)
        inp = torch.from_numpy(inp)
        inp = torch.unsqueeze(inp, 0).float()
        model = U_Net()
        ckpt = load_checkpoint('SegmentModel/VesselSeg_UNet3d_base_02-27.pth')
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        #model = NetCustom(model)

#intr= Interpretability(resModel)

intr= Interpretability(model)
expln = Explainability(model)

transform_func =  intr.inpTransformSetup(config['resize'], config['centre_crop'], config['mean_vec'], config['std_vec'] )
warnings.filterwarnings("ignore")

data_transform = transforms.Compose([
        transforms.Resize(256),                            
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225])
])

#wrapper = intr.wrap
wrapper = intr.binaraySegWrap
#torch.argmax(resnet18(torch.unsqueeze(data_transform(classigfication_img), 0)))
if not os.path.exists('Log'):
  os.mkdir('Log')
  logging.info("Log filepath created")

logging.basicConfig(filename="Log/AppLog.log", level=getattr(logging, config['log_level'].upper(), None),
                    format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

logging.info("##################################################")
logging.info("Process starting")

filePath = config['model_name'] + '_' + config['dataset'] +'_' + config['test_run']
if not os.path.exists(filePath):
  os.mkdir(filePath)
  logging.info("Output filepath created")


if config['model_nature'] == 'Segmentation':
  logging.info("Running segmentation task.")
  for method in tqdm(list(filter(lambda x: x['use']== True, config['methods'])), desc="Processing...."):
      if method['use']:
          try:
              logging.info("Running function: "+method['method_name'] )
              getattr(intr, method['method_name'])(model, method['target'], inp, method['inpt_transform_req'],
                                                 transform_func, method['aux_func_flag'], wrapper, method['visualize_method'],
                                                 method['sign'], method['show_colorbar'], method['title'], method['device_id'],
                                                 filePath, inp_path.split('/')[-1].split('.')[0], config['is_3d'],config['isDepthFirst'],**method['keywords'] )
          except Exception:
              logging.error("Unexpected error: " + method['method_name'])
              logging.error(sys.exc_info()[1])

  for method in tqdm(list(filter(lambda x: x['use']== True, config['explain_methods'])), desc="Processing...."):
      if method['use']:
          try:
              trans_func = None
              if method["library"]=='lime':
                  trans_func = getattr(expln, "inpTransformSetupLime")(config['resize'], config['centre_crop'], config['mean_vec'], config['std_vec'])
              generic_trans_func = getattr(expln, "inpTransformSetup")(config['resize'], config['centre_crop'], config['mean_vec'], config['std_vec'])
              logging.info("Running explainability function: "+method['method_name'] )
              getattr(expln, method['method_name'])(model, inp, method['inpt_transform_req'],
                                                 generic_trans_func, method['aux_func_flag'], wrapper, method['device_id'],
                                                 filePath, inp_path.split('/')[-1].split('.')[0], trans_func, config['is_3d'],**method['keywords'] )
          except Exception:
              logging.error("Unexpected error: " + method['method_name'])
              logging.error(sys.exc_info()[1])

if config['model_nature'] == 'Classification':
  logging.info("Running classification task.")
  for method in tqdm(list(filter(lambda x: x['use']== True, config['methods'])), desc="Processing...."):
      if method['use']:
          try:
              logging.info("Running function: "+ method['method_name'])
              getattr(intr, method['method_name'])(model, method['target'],
                                                 inp, method['inpt_transform_req'], transform_func, method['aux_func_flag'],
                                                 wrapper, method['visualize_method'], method['sign'], method['show_colorbar'], method['title'],
                                                 method['device_id'], filePath, inp_path.split('/')[-1].split('.')[0],config['is_3d'],config['isDepthFirst'],
                                                 **method['keywords']  )

          except Exception:
              logging.error("Unexpected error: "+ method['method_name'])
              logging.error(sys.exc_info()[1])
  for method in tqdm(list(filter(lambda x: x['use']== True, config['explain_methods'])), desc="Processing...."):
      if method['use']:
          try:
              trans_func = None
              if method["library"]=='lime':
                  trans_func = getattr(expln, "inpTransformSetupLime")(config['resize'], config['centre_crop'], config['mean_vec'], config['std_vec'])
              generic_trans_func = getattr(expln, "inpTransformSetup")(config['resize'], config['centre_crop'], config['mean_vec'], config['std_vec'])
              logging.info("Running explainability function: "+method['method_name'] )
              getattr(expln, method['method_name'])(model, inp, method['inpt_transform_req'],
                                                 generic_trans_func, method['aux_func_flag'], wrapper, method['device_id'],
                                                 filePath, inp_path.split('/')[-1].split('.')[0], trans_func, config['is_3d'],**method['keywords'] )
          except Exception:
              logging.error("Unexpected error: " + method['method_name'])
              logging.error(sys.exc_info()[1])


logging.info("Process finished")
logging.info("##################################################")