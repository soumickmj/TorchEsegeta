import copy
import json
from pipeline import Pipeline
import multiprocessing.dummy as multiprocessing
import matplotlib.pyplot as plt
plt.switch_backend('agg')

"""try:
    set_start_method('spawn',  force=True)
except:
    pass"""

class Pipeline_multi_process:
    def __init__(self, model = None, jsonpath= None , chekpoint = None ):
        f = open(jsonpath)
        config = json.load(f)
        f.close()
        unique_device_list = set(map(lambda method: method['device_id'], config['methods']))
        self.config_list = []
        self.pipeline_obj_list = []
        for uniq_device in unique_device_list:
            temp_config = copy.deepcopy(config)
            temp_config['methods'] = list(filter(lambda method: method['device_id'] == uniq_device, config['methods']))
            if temp_config['explain_methods'][0]['device_id'] != uniq_device: temp_config['explain_methods'] = [] #temp_config.pop('explain_methods', None)
            #self.config_list.append(temp_config)
            self.config_list += self.share_gpu_split(temp_config)

        count = 1
        for configs in self.config_list:
            self.pipeline_obj_list.append(Pipeline(model= model, jsonpath=configs, chekpoint= chekpoint, instance_id=count))
            count += 1

    def interpret_explain(self, **kwargs):
        jobs = []
        for i in range(len(self.pipeline_obj_list)):
            p = multiprocessing.Process(target=self.pipeline_obj_list[i].interpret_explain, kwargs=kwargs)
            jobs.append(p)
            p.start()

    def share_gpu_split(self, config):
        config['methods'] = list(filter(lambda method: method['use'] == True, config['methods']))
        n_methods = len(config['methods'])
        if n_methods == 0:
            return []
        elif n_methods > 1 and config["share_gpu_threads"] > 1:
            gpu_sharing_methods = len(
                list(filter(lambda method: 'share_gpu' in method and method['share_gpu'] == True, config['methods'])))
            if n_methods != gpu_sharing_methods:  # Even if one method doesn't agree with gpu sharing, then gpu sharing won't be used
                return [config]
            config_per_threads = []
            methods_per_thread = n_methods // config["share_gpu_threads"]
            for i in range(config["share_gpu_threads"]):
                temp_config = copy.deepcopy(config)
                if i + 1 == config["share_gpu_threads"]:  # final thread
                    temp_config['methods'] = temp_config['methods'][(i * methods_per_thread):]
                else:
                    temp_config['methods'] = temp_config['methods'][
                                             (i * methods_per_thread):(i * methods_per_thread) + methods_per_thread]
                config_per_threads.append(temp_config)
            return config_per_threads
        else:
            return [config]


if __name__ == '__main__':
    pl = Pipeline_multi_process(jsonpath='cfg_Misc_Seg_3d.json')
    pl.interpret_explain(tag='myOutput',target = [1],output_path="/project/ardas")