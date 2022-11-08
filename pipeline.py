import os
import time
from scipy.ndimage import zoom
from torchvision import models
from methods import Interpretability
from explainMethods import Explainability
from model_copy import Net, Bottleneck, R2Plus1dStem4MRI, modifybasicstem
from PIL import Image
import json
import torch
import logging
import warnings
from tqdm import tqdm
from torch import nn
import numpy as np
import pandas as pd
# from NiftyFilerLoader import R2Plus1dStem4MRI, load_checkpoint
import nibabel as nb
import torchvision.transforms as transforms
from SegmentModel.VesselSeg_UNet3d import U_Net, U_Net_DeepSup
from segmentWrapper import SegmentWrapper, MultiClassSegmentWrapper
from utils import load_checkpoint, get_logger
from uncertainity import cascading_randomization
import sys
import signal


class Pipeline:
    def __init__(self, model = None, jsonpath= None , chekpoint = None, instance_id = 1 ):
        self.model = model
        self.jsonpath = jsonpath
        self.inp = None
        self.inp_path = None
        self.instance_id = instance_id
        self.wrapper_dict = {
            'threshold_based' : SegmentWrapper,
            'multi_class' : MultiClassSegmentWrapper
        }
        self.uncertainity_metrics = []
        self.uncertainity_metrics_CasRand = []
        if chekpoint is not None:
            ckpt = load_checkpoint(chekpoint)
            self.model.load_state_dict(ckpt['state_dict'])
            self.model.eval()
        if type(jsonpath).__name__ == 'dict':
            self.config = jsonpath
        else:
            with open(self.jsonpath) as f:
                self.config = json.load(f)
        log_folder_created = False
        if not os.path.exists('Log'):
            os.makedirs('Log', exist_ok = True)
            log_folder_created = True


        self.logger = get_logger(
            f"TorchEsegeta_{str(self.instance_id)}",
            f"Log/AppLog_thread_{str(self.instance_id)}.log",
            getattr(logging, self.config['log_level'].upper(), None),
        )

        if log_folder_created:
            self.logger.info("Log filepath created")

        if self.config["timeout_enabled"]:
            try:
                signal.signal(signal.SIGALRM, self.handler)
                signal.alarm(self.config["timeout_time"])
            except Exception:
                self.logger.warn("Timeout funtionality only works for Linux")

        if (self.model is None or self.config["default"] == True) and self.config[
            "default"
        ]:
            if not self.config["is_3d"]:
                if self.config['model_nature'] == 'Segmentation':
                    self.inp_path = 'testImage.jpg'
                    self.inp = Image.open(self.inp_path)
                    resModel = models.segmentation.fcn_resnet101(pretrained=True)
                    # print(resModel.state_dict().keys())
                    self.model = Net(resModel)
                    self.model = MultiClassSegmentWrapper(self.model)
                    self.model.eval()
                if self.config['model_nature'] == 'Classification':
                    self.model = models.resnet18(pretrained=True)
                    self.inp_path = 'imagenetImage.jpg'
                    self.inp = Image.open(self.inp_path)
                    self.model.eval()
            else:
                if self.config['model_nature'] == 'Classification':
                    self.inp_path = 'BraTS19_2013_0_1_t1ce.nii.gz'
                    self.inp = nb.load(self.inp_path)
                    self.inp = self.inp.get_fdata()
                    self.inp = self.inp.squeeze()
                    if self.config["patch_size"] == -1:
                        self.inp = zoom(self.inp, (0.5, 0.5, 0.5))
                    #self.inp = (self.inp - np.min(self.inp)) / (np.max(self.inp) - np.min(self.inp))
                    self.inp = np.moveaxis(self.inp, -1, 0)
                    self.inp = torch.from_numpy(self.inp)
                    self.inp = torch.unsqueeze(self.inp, 0).float()
                    self.model = models.video.r3d_18(pretrained=True)
                    #self.model.stem = R2Plus1dStem4MRI()
                    self.model.stem = modifybasicstem()
                    self.model.fc = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(self.model.fc.in_features, 3)
                    )

                    ckpt = load_checkpoint('resnet3d.pth.tar')
                    self.model.load_state_dict(ckpt['state_dict'])
                    self.model.eval()
                if self.config['model_nature'] == 'Segmentation':
                    self.inp_path = 'SegmentModel/data/SegTestVol.nii'
                    # self.inp_path = 'SegmentModel/data/IXI425-IOP-0988-MRA.nii.gz'
                    self.inp = nb.load(self.inp_path)
                    self.inp = self.inp.get_fdata()
                    self.inp = self.inp.squeeze()
                    if self.config["patch_size"] == -1:
                        # self.inp = zoom(self.inp, (0.125, 0.125, 0.35))
                        self.inp = zoom(self.inp, (0.089, 0.101, 0.33))
                    self.inp = (self.inp - np.min(self.inp)) / (np.max(self.inp) - np.min(self.inp))
                    # print(self.inp.shape)
                    # self.inp = self.inp.squeeze()
                    self.inp = np.moveaxis(self.inp, -1, 0)
                    # print(self.inp.shape)
                    self.inp = np.moveaxis(self.inp, -1, -2)
                    # print(self.inp.shape)
                    self.inp = torch.from_numpy(self.inp)
                    self.inp = torch.unsqueeze(self.inp, 0).float()
                    # print(self.inp.shape)
                    self.model = U_Net()
                    ckpt = load_checkpoint('SegmentModel/checkpointbestUNet.pth')
                    # ckpt = load_checkpoint('SegmentModel/checkpointbestUNetMSS_Deform.pth')
                    # ckpt = load_checkpoint('SegmentModel/checkpointbest50UNetMSS.pth')
                    self.model.load_state_dict(ckpt['state_dict'])
                    self.model = SegmentWrapper(self.model)
                    self.model.eval()
                        # print("loaded")

    def handler(self, signum, frame):
        if self.config["timeout_enabled"]:
            signal.alarm(self.config["timeout_time"])
        raise Exception("Timeout!!! Preset wait time elapsed.")



    def interpret_explain(self,tag,inp= None, target= None, output_path= None, model_name=None, wrapper_fnction='threshold_based'):
        if self.inp is not None:
            inp = self.inp
        if not self.config['default'] and self.config['model_nature'] == 'Segmentation':
            if wrapper_fnction not in self.wrapper_dict.keys():
                raise Exception("Supplied wrapper function parameter '%s' is not a valid one." %wrapper_fnction)
            model = self.wrapper_dict[wrapper_fnction](self.model)
        else:
            model = self.model


        available_wrapper_classes = list(map(lambda wrapper_class : wrapper_class.__name__, self.wrapper_dict.values()))

        patch_size= self.config["patch_size"] if self.config["patch_size"] is not None else None
        patch_overlap = self.config["patch_overlap"] if self.config["patch_overlap"] is not None else None
        intr = Interpretability(model, available_wrapper_classes, patch_size, patch_overlap, self.config['amp_enbled'])
        expln = Explainability(model)

        transform_func = intr.inpTransformSetup(self.config['resize'], self.config['centre_crop'], self.config['mean_vec'],
                                                self.config['std_vec'])
        warnings.filterwarnings("ignore")



        #logging.basicConfig(filename="Log/AppLog.log", level=getattr(logging, self.config['log_level'].upper(), None),
        #                    format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

        self.logger.info("##################################################")
        self.logger.info("Process starting")



        filePath = model_name if model_name is not None else self.config['model_name'] + '_' +\
                        self.config['dataset'] + '_' + self.config['test_run']
        filePath = os.path.join(output_path if output_path is not None else self.config["output_path"],
                                filePath)
        if not os.path.exists(filePath):
            os.makedirs(filePath, exist_ok=True)
            self.logger.info("Output filepath created")

        if self.config['model_nature'] == 'Segmentation':
            self.logger.info("Running segmentation task.")
            for method in tqdm(list(filter(lambda x: x['use'] == True, self.config['methods'])), desc="Processing...."):
                if method['use']:

                     try:
                        self.logger.info("Running function: " + method['method_name'])
                        uncertain_metrics = getattr(intr, method['method_name'])(model, target, inp, method['inpt_transform_req'],
                                                             transform_func,
                                                             method['uncertainity_metrics'],
                                                             method['visualize_method'],
                                                             method['sign'], method['show_colorbar'], method['title'],
                                                             method['device_id'],
                                                             method['library'],
                                                            filePath, tag,
                                                            self.config['is_3d'],
                                                            self.config['isDepthFirst'],
                                                            method['patch_required'],
                                                            self.config['batch_dim_present'],
                                                             **method['keywords'])




                        if uncertain_metrics is not None: self.uncertainity_metrics.append(uncertain_metrics)


                        if method['uncertainity_cascading'] in [1,2]:

                            uncertain_metrics_casRand = cascading_randomization(intr, method = method['method_name'],forward_function = model, target =  target, inp_image= inp, inp_transform_flag= method['inpt_transform_req'],
                                                             transform_func =transform_func,
                                                 uncertainity_mode = method['uncertainity_cascading'],
                                                 visualize_method = method['visualize_method'],
                                                 sign = method['sign'], show_colorbar = method['show_colorbar'], title = method['title'],
                                                 device = method['device_id'],
                                                 library = method['library'],
                                                 output_path = filePath, input_file = tag,
                                                 is_3d = self.config['is_3d'], isDepthFirst = self.config['isDepthFirst'], patcher_flag = method['patch_required'],
                                                 batch_dim_present = self.config['batch_dim_present'],
                                                             **method['keywords'])

                            self.uncertainity_metrics_CasRand.append(uncertain_metrics_casRand)

                     except Exception:
                         self.logger.error("Unexpected error: " + method['method_name'])
                         self.logger.error(sys.exc_info()[1])

            for method in tqdm(list(filter(lambda x: x['use'] == True, self.config['explain_methods'])),
                               desc="Processing...."):
                if method['use']:
                    try:
                        trans_func = None
                        if method["library"] == 'lime':
                            trans_func = getattr(expln, "inpTransformSetupLime")(self.config['resize'],
                                                                                 self.config['centre_crop'],
                                                                                 self.config['mean_vec'], self.config['std_vec'])
                        generic_trans_func = getattr(expln, "inpTransformSetup")(self.config['resize'],
                                                                                 self.config['centre_crop'],
                                                                                 self.config['mean_vec'], self.config['std_vec'])
                        self.logger.info("Running explainability function: " + method['method_name'])
                        getattr(expln, method['method_name'])(model, inp, method['inpt_transform_req'],
                                                              generic_trans_func,
                                                              method['device_id'],
                                                              filePath, tag,
                                                              trans_func, self.config['is_3d'],target,batch_dim_present = self.config['batch_dim_present'],
                                                              **method['keywords'])

                    except Exception:
                        self.logger.error("Unexpected error: " + method['method_name'])
                        self.logger.error(sys.exc_info()[1])

        if self.config['model_nature'] == 'Classification':
            self.logger.info("Running classification task.")
            for method in tqdm(list(filter(lambda x: x['use'] == True, self.config['methods'])), desc="Processing...."):
                if method['use']:
                     try:
                        self.logger.info("Running function: " + method['method_name'])
                        uncertain_metrics = getattr(intr, method['method_name'])(model, target,
                                                             inp, method['inpt_transform_req'], transform_func,
                                                             method['uncertainity_metrics'],
                                                             method['visualize_method'], method['sign'],
                                                             method['show_colorbar'], method['title'],
                                                             method['device_id'],
                                                             method['library'], filePath,
                                                             tag, self.config['is_3d'],
                                                             self.config['isDepthFirst'],method['patch_required'],
                                                             self.config['batch_dim_present'],
                                                             **method['keywords'])
                        if uncertain_metrics is not None:  self.uncertainity_metrics.append(uncertain_metrics)

                        if method['uncertainity_cascading'] in [1, 2]:

                            uncertain_metrics_casRand = cascading_randomization(intr, method=method['method_name'],
                                                                        forward_function=model, target=target,
                                                                        inp_image=inp,
                                                                        inp_transform_flag=method['inpt_transform_req'],
                                                                        transform_func=transform_func,
                                                                        uncertainity_mode=method[
                                                                            'uncertainity_cascading'],
                                                                        visualize_method=method['visualize_method'],
                                                                        sign=method['sign'],
                                                                        show_colorbar=method['show_colorbar'],
                                                                        title=method['title'],
                                                                        device=method['device_id'],
                                                                        library=method['library'],
                                                                        output_path=filePath, input_file=tag,
                                                                        is_3d=self.config['is_3d'],
                                                                        isDepthFirst=self.config['isDepthFirst'],
                                                                        patcher_flag=method['patch_required'],
                                                                        batch_dim_present=self.config[
                                                                            'batch_dim_present'],
                                                                        **method['keywords'])

                            self.uncertainity_metrics_CasRand.append(uncertain_metrics_casRand)

                     except Exception:
                        self.logger.error("Unexpected error: " + method['method_name'])
                        self.logger.error(sys.exc_info()[1])
            for method in tqdm(list(filter(lambda x: x['use'] == True, self.config['explain_methods'])),
                               desc="Processing...."):
                if method['use']:
                    try:
                        trans_func = None
                        if method["library"] == 'lime':
                            trans_func = getattr(expln, "inpTransformSetupLime")(self.config['resize'],
                                                                                 self.config['centre_crop'],
                                                                                 self.config['mean_vec'], self.config['std_vec'])
                        generic_trans_func = getattr(expln, "inpTransformSetup")(self.config['resize'],
                                                                                 self.config['centre_crop'],
                                                                                 self.config['mean_vec'], self.config['std_vec'])
                        self.logger.info("Running explainability function: " + method['method_name'])
                        getattr(expln, method['method_name'])(model, inp, method['inpt_transform_req'],
                                                              generic_trans_func,
                                                              method['device_id'],
                                                              filePath, tag,
                                                              trans_func, self.config['is_3d'],target, batch_dim_present = self.config['batch_dim_present'],
                                                              **method['keywords'])
                    except Exception:
                        self.logger.error("Unexpected error: " + method['method_name'])
                        self.logger.error(sys.exc_info()[1])

        # savecsv

        if len(self.uncertainity_metrics) > 0:
            df_um = pd.DataFrame(self.uncertainity_metrics)
            if not df_um.empty:
                df_um.to_csv(filePath+"/"+"uncertainity_metrics"+time.strftime("%Y%m%d-%H%M%S")+".csv", header= False)


        df_um_cr = pd.DataFrame(self.uncertainity_metrics_CasRand)
        if not df_um_cr.empty:
            df_um_cr.to_csv(filePath+"/" + "uncertainity_metrics_CasRand"+ time.strftime("%Y%m%d-%H%M%S")+".csv", header= False)



        self.logger.info("Process finished")
        self.logger.info("##################################################")