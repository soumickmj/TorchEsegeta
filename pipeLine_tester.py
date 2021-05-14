import os
from pipeline import Pipeline
from pipeline_wrapper import Pipeline_multi_process

# http settings related to cluster
os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'

pl = Pipeline(jsonpath='cfg_Misc_class_3d.json')
pl.interpret_explain(tag='FinalTestOutput',target = [0],output_path="results/3dClasification_LGG")