import json
import PySimpleGUI as sg

disp = True


def HeaderParams(path):
    sg.theme('Default1')
    sg.SetOptions(text_justification='right')
    with open(path) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    parameters = []
    for k in jsonObject:
        if k != "methods" and k != "explain_methods":
            parameters.append(k)
    parms = [[sg.Text('model_nature', size=(15, 1)),
              sg.Drop(values=('Segmentation', 'Classification'), default_value=jsonObject['model_nature'],
                      auto_size_text=True),
              sg.Text('model_name', size=(15, 1)),
              sg.Drop(values=('Vessel_seg_UNET', 'Vessel_seg_UNET'), default_value=jsonObject['model_name'],
                      auto_size_text=True), ],
             [sg.Text('is_3d', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(jsonObject['is_3d']), auto_size_text=True),
              sg.Text('isDepthFirst', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(jsonObject['isDepthFirst']), auto_size_text=True), ],
             [sg.Text('batch_dim_present', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(jsonObject['batch_dim_present']),
                      auto_size_text=True),
              sg.Text('default', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(jsonObject['default']), auto_size_text=True), ],
             [sg.Text('dataset', size=(15, 1)),
              sg.Drop(values=('IXI_MRA', 'dataset_2'), default_value=jsonObject['dataset'], auto_size_text=True),
              sg.Text('test_run', size=(15, 1)),
              sg.Drop(values=('final_test', 'test'), default_value=jsonObject['test_run'], auto_size_text=True), ],
             [sg.Text('resize', size=(15, 1)), sg.In(default_text=jsonObject['resize'], size=(8, 1)),
              sg.Text('centre_crop', size=(15, 1)), sg.In(default_text=jsonObject['centre_crop'], size=(8, 1)), ],
             [sg.Text('mean_vec', size=(15, 1)), sg.In(default_text=str(jsonObject['mean_vec']), size=(8, 1)),
              sg.Text('std_vec', size=(15, 1)), sg.In(default_text=str(jsonObject['std_vec']), size=(8, 1)), ],
             [sg.Text('patch_overlap', size=(15, 1)), sg.In(default_text=str(jsonObject['patch_overlap']), size=(8, 1)),
              sg.Text('_comment', size=(15, 1)), sg.In(default_text=str(jsonObject['_comment']), size=(8, 1)), ],
             [sg.Text('patch_size', size=(15, 1)), sg.In(default_text=str(jsonObject['patch_size']), size=(8, 1)),
              sg.Text('log_level', size=(15, 1)), sg.In(default_text=str(jsonObject['log_level']), size=(8, 1)), ],
             [sg.Text('output_path', size=(15, 1)), sg.In(default_text=str(jsonObject['output_path']), size=(8, 1)),
              sg.Text('amp_enbled', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(jsonObject['amp_enbled']), auto_size_text=True), ],
             [sg.Text('share_gpu_threads', size=(15, 1)),
              sg.In(default_text=str(jsonObject['share_gpu_threads']), size=(8, 1)),
              sg.Text('timeout_enabled', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(jsonObject['timeout_enabled']),
                      auto_size_text=True), ],
             ]

    layout = [[sg.Frame('Parameters', parms, title_color='black', font='Any 12')],
              [sg.Submit(size=(15, 1), button_text="Select Methods"), sg.Cancel(size=(15, 1))]]
    window = sg.Window('Torch Esgeta', font=("Helvetica", 12)).Layout(layout)
    button, values = window.Read()
    sg.SetOptions(text_justification='left')
    if disp:
        print(button, values)

    param_dict = dict(zip(parameters, values.values()))
    param_dict.update({"methods": "0"})

    return param_dict, jsonObject


def MethodsSelector():
    sg.theme('Default1')
    sg.SetOptions(text_justification='right')

    Captum = [[sg.Checkbox('Saliency', size=(20, 1)),
               sg.Checkbox('Feature Ablation', size=(20, 1)),
               sg.Checkbox('IG', size=(20, 1)),
               sg.Checkbox('Occlusion', size=(20, 1)),
               sg.Checkbox('layer_conductance', size=(20, 1)), ],

              [sg.Checkbox('GBP', size=(20, 1)),
               sg.Checkbox('Deep Lift', size=(20, 1)),
               sg.Checkbox('Deconv', size=(20, 1)),
               sg.Checkbox('Guided GradCam', size=(20, 1)),
               sg.Checkbox('gradient_shap', size=(20, 1)),
               ],
              [
                  sg.Checkbox('Layer Act', size=(20, 1)),
                  sg.Checkbox('internal_influence', size=(20, 1)),
                  sg.Checkbox('layer_grad_shap', size=(20, 1)),
                  sg.Checkbox('input_x_gradient', size=(20, 1)),
                  sg.Checkbox('ShapleyValueSampling', size=(20, 1)),
              ],
              [
                  sg.Checkbox('deep_lift_shap', size=(20, 1)),
                  sg.Checkbox('LayerGradCam', size=(20, 1)),
                  sg.Checkbox('LayerGradientXActivation', size=(20, 1)),
                  sg.Checkbox('LayerDeepLift', size=(20, 1)), ], ]

    Other = [[sg.Checkbox('lucent_render_vis', size=(20, 1)),
              sg.Checkbox('lime_segmentation', size=(20, 1)), ]]

    CNN = [[sg.Checkbox('score_cam', size=(20, 1)),
            sg.Checkbox('vanilla_backprop', size=(20, 1)),
            sg.Checkbox('layer_activation_GBP', size=(20, 1)),
            sg.Checkbox('grad_cam', size=(20, 1)),
            sg.Checkbox('layer_visualization', size=(20, 1)), ],

           [sg.Checkbox('grad_times_image', size=(20, 1)),
            sg.Checkbox('DeepDream', size=(20, 1)),
            sg.Checkbox('GBP', size=(20, 1)),
            sg.Checkbox('IG', size=(20, 1)),
            sg.Checkbox('GGCam', size=(20, 1)), ]
           ]

    Torchray = [[sg.Checkbox('Deconv', size=(20, 1)),
                 sg.Checkbox('Gradcam', size=(20, 1)),
                 sg.Checkbox('Gradient', size=(20, 1)),
                 sg.Checkbox('Excitation Backprop', size=(20, 1)), ],
                [sg.Checkbox('Guidedbackprop', size=(20, 1)),
                 sg.Checkbox('Contrast Exct Bacp', size=(20, 1)),
                 sg.Checkbox('Linear Approximation', size=(20, 1))]]

    layout = [[sg.Frame('Captum', Captum, font='Any 12', title_color='black')],
              [sg.Frame('CNN', CNN, font='Any 12', title_color='black')],
              [sg.Frame('Torchray', Torchray, font='Any 12', title_color='black')],
              [sg.Frame('Other', Other, font='Any 12', title_color='black')],
              [sg.Submit(size=(15, 1), button_text="Update Fields"), sg.Cancel(size=(15, 1))]]

    methods = ['Saliency', 'Feature_Ablation', 'Integrated_Gradients', 'Occlusion', 'layer_conductance',
               'Guided_Backprop', 'Deep_Lift',
               'Deconvolution', 'Guided_Grad_Cam', 'gradient_shap', 'Layer_Activation', 'internal_influence',
               'layer_grad_shap',
               'input_x_gradient', 'ShapleyValueSampling', 'deep_lift_shap', 'LayerGradCam', 'LayerGradientXActivation',
               'LayerDeepLift', 'score_cam', 'vanilla_backprop', 'layer_activation_guided_backprop', 'grad_cam',
               'layer_visualization',
               'grad_times_image', 'DeepDream', 'guided_backprop', 'integrated_grad', 'guided_grad_cam', 'Deconv',
               'Gradcam', 'Gradient',
               'contrast_excitation_backprop', 'Guidedbackprop', 'contrast_excitation_backprop', 'Linearapprox',
               'lucent_render_vis', 'segmentation']
    window = sg.Window('Torch Esgeta - Methods selector', font=("Helvetica", 12)).Layout(layout)
    button, values = window.Read()
    sg.SetOptions(text_justification='left')
    if disp:
        print(button, values)
    method_names = []
    for i in values:
        if values[i]:
            if 0 <= i <= 18:
                # print(methods[i] + "_captum")
                method_names.append(methods[i] + "_captum")
            elif 19 <= i <= 28:
                # print(methods[i] + "_CNN Visualization")
                method_names.append(methods[i] + "_CNN Visualization")
            elif 29 <= i <= 35:
                # print(methods[i] + "_torchray")
                method_names.append(methods[i] + "_torchray")
            elif i == 36:
                # print(methods[i] + "_lucent")
                method_names.append(methods[i] + "_lucent")
            elif i == 37:
                # print(methods[i] + "_lime")
                method_names.append(methods[i] + "_lime")
    return method_names


def loadParams(method_name, library_name, path):
    with open(path) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
        methods = jsonObject["methods"]
    parameters = ""
    for k in range(len(methods)):
        if method_name.lower() in methods[k]["method_name"].lower() and methods[k]["library"] == library_name:
            parameters = methods[k]
            break
    parms = [[sg.Text('inpt_transform_req', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(parameters['inpt_transform_req']),
                      auto_size_text=True),
              sg.Text('visualize_method', size=(15, 1)),
              sg.Drop(values=('heat_map', 'heat_map'), default_value=parameters['visualize_method'],
                      auto_size_text=True), ],
             [sg.Text('sign', size=(15, 1)),
              sg.Drop(values=('positive', 'negative'), default_value=str(parameters['sign']), auto_size_text=True),
              sg.Text('show_colorbar', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(parameters['show_colorbar']), auto_size_text=True), ],
             [sg.Text('device_id', size=(15, 1)), sg.In(default_text=parameters['device_id'], size=(8, 1)),
              sg.Text('share_gpu', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(parameters['share_gpu']), auto_size_text=True), ],
             [sg.Text('patch_required', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(parameters['patch_required']), auto_size_text=True),
              sg.Text('use', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(parameters['use']), auto_size_text=True), ],
             [sg.Text('uncertainity_metrics', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(parameters['uncertainity_metrics']),
                      auto_size_text=True),
              sg.Text('uncertainity_cascading', size=(15, 1)),
              sg.In(default_text=parameters['uncertainity_cascading'], size=(8, 1)), ],
             [sg.Text('uncertainity', size=(15, 1)),
              sg.Drop(values=('true', 'false'), default_value=str(parameters['uncertainity']), auto_size_text=True),
              sg.Text('keywords', size=(15, 1)), sg.In(default_text=str(parameters['keywords']), size=(8, 1)), ]]

    text = 'Parameteres for - ' + str(method_name)
    layout = [[sg.Frame(text, parms, title_color='black', font='Any 12')],
              [sg.Submit(size=(15, 1), button_text="Generate JSON"), sg.Submit(size=(20, 1), button_text="Next"),
               sg.Cancel(size=(15, 1))]]
    window = sg.Window(str(method_name), font=("Helvetica", 12)).Layout(layout)
    button, values = window.Read()
    if disp:
        print("#######################################################################################\n")
        print(button)

    sg.SetOptions(text_justification='left')
    flag = 0
    if button == "Generate JSON":
        if disp:
            print("inside JSON Creator")
            print(parameters)
        flag = 1
        return parameters, flag
    else:
        # print(button, values)
        parameters['inpt_transform_req'] = values[0]
        parameters['visualize_method'] = values[1]
        parameters['sign'] = values[2]
        parameters['show_colorbar'] = values[3]
        parameters['device_id'] = values[4]
        parameters['share_gpu'] = values[5]
        parameters['patch_required'] = values[6]
        parameters['use'] = values[7]
        parameters['uncertainity_metrics'] = values[8]
        parameters['uncertainity_cascading'] = values[9]
        parameters['uncertainity'] = values[10]
        parameters['keywords'] = values[11]
        # print(parameters)

        for k in range(len(methods)):
            if method_name.lower() in methods[k]["method_name"].lower() and methods[k]["library"] == library_name:
                if disp:
                    print("Before Update:", methods[k])
                methods[k] = parameters
                if disp:
                    print("After Update:", methods[k])
                break
        if disp:
            print(methods)
    return parameters, flag, methods


def main(JsonFilePath):
    param_dict, jsonObject = HeaderParams(JsonFilePath)
    if disp:
        print("########################### V1 - START ###########################")
        print(jsonObject)
        print("########################### V1 - END ###########################")
        print("\n\n")

        print("########################### V2 - START ###########################")
    values = MethodsSelector()
    method = []
    library = []
    for i in range(len(values)):
        library_name = values[i].split("_")[-1]
        library.append(library_name)
        library_name = "_" + library_name
        method_name = values[i].replace(library_name, "")
        method.append(method_name)
    if disp:
        print("########################### V2 - END ###########################")
        print("\n\n")
        print("########################### V3 - START ###########################")
    for i in range(len(values)):
        if i == 0:
            path = "Base.json"
        else:
            path = "Output.json"
        a, flag, jsonFile = loadParams(method[i], library[i], path)
        if flag:
            break
        param_dict["methods"] = jsonFile
        with open('Output.json', 'w') as f:
            json.dump(param_dict, f, indent=5)
    if disp:
        print("########################### V3 - END ###########################")


if __name__ == "__main__":
    jsonFilePath = "C:/Users/budha/PycharmProjects/Esgeta_GUI/jsons/TorchEsegeta_cfg_Misc_Seg_3d.json"
    main(jsonFilePath)
