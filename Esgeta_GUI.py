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

    l = 30
    feature_based = [
        [
            sg.Checkbox('Shapley Value Sampling [captum]', size=(l, 1)),
            sg.Checkbox('Feature Ablation [captum]', size=(l, 1)),
            sg.Checkbox('Occlusion [captum]', size=(l, 1)),
            sg.Checkbox('Score cam [CNN Vis]', size=(l, 1)),
        ],
    ]
    gradient_based = [
        [

            sg.Checkbox('Guided Backprop [CNN Vis]', size=(l, 1)),
            sg.Checkbox('Guided Backprop [captum]', size=(l, 1)),
            sg.Checkbox('Guided Backprop  [torchray]', size=(l, 1)),
            sg.Checkbox('Saliency [captum]', size=(l, 1)),

        ],
        [
            sg.Checkbox('Input x gradient [captum]', size=(l, 1)),
            sg.Checkbox('Integrated Grad [CNN Vis]', size=(l, 1)),
            sg.Checkbox('Deconvolution [captum]', size=(l, 1)),
            sg.Checkbox('Deconvolution [torchray]', size=(l, 1)),
        ],
        [
            sg.Checkbox('Grad times image [CNN Vis]', size=(l, 1)),
            sg.Checkbox('Deep Lift [captum]', size=(l, 1)),
            sg.Checkbox('Deep Lift Shap [captum]', size=(l, 1)),
            sg.Checkbox('Integrated Gradients [captum]', size=(l, 1)),
        ],
        [
            sg.Checkbox('Gradient Shap [captum]', size=(l, 1)),
            sg.Checkbox('Guided Grad Cam [captum]', size=(l, 1)),
            sg.Checkbox('Guided Grad Cam [CNN Vis]', size=(l, 1)),
            sg.Checkbox('Vanilla Backprop [CNN Vis]', size=(l, 1)),
        ],
        [
            sg.Checkbox('Layer Visualization [CNN Vis]', size=(l, 1)),
        ]
    ]

    Interpretability_Model_Attribution = [
        [sg.Frame('Feature Based', feature_based, font='Any 12', title_color='black')],
        [sg.Frame('Gradient Based', gradient_based, font='Any 12', title_color='black')]

    ]

    Interpretability_Layer_Attribution = [
        [
            sg.Checkbox('Layer Conductance [captum]', size=(l, 1)),
            sg.Checkbox('Layer Activation [captum]', size=(l, 1)),
            sg.Checkbox('Internal Influence [captum]', size=(l, 1)),
            sg.Checkbox('Layer Grad Shap [captum]', size=(l, 1)),

        ],
        [
            sg.Checkbox('Layer DeepLift [captum]', size=(l, 1)),
            sg.Checkbox('Layer Activation GBP[CNN Vis]', size=(l, 1)),
            sg.Checkbox('Layer GradCam [captum]', size=(l, 1)),
            sg.Checkbox('Grad Cam [CNN Vis]', size=(l, 1)),
        ],
        [
            sg.Checkbox('LayerGradientXActivation [captum]', size=(l, 1)),
            sg.Checkbox('Grad Cam [torchray]', size=(l, 1)),
            sg.Checkbox('Gradient [torchray]', size=(l, 1)),
            sg.Checkbox('Contrast Excitation Backprop [torchray]', size=(l, 1)),

        ],
        [
            sg.Checkbox('Contrast Excitation Backprop [torchray]', size=(l, 1)),
            sg.Checkbox('Linear Approx [torchray]', size=(l, 1)),
        ]
    ]

    Explainability = [
        [
            sg.Checkbox('Render Visualization [lucent]', size=(l, 1)),
            sg.Checkbox('Segmentation [lime]', size=(l, 1)),
            sg.Checkbox('DeepDream [CNN Vis]', size=(l, 1)),

        ],
    ]

    layout = [
        [sg.Frame('Interpretability : Model Attribution', Interpretability_Model_Attribution, font='Any 12', title_color='black',)],
        [sg.Frame('Interpretability : Layer Attribution', Interpretability_Layer_Attribution, font='Any 12', title_color='black')],
        [sg.Frame('Explainability', Explainability, font='Any 12', title_color='black')],
        [sg.Submit(size=(20, 1), button_text="Update Fields"), sg.Cancel(size=(20, 1))]
    ]

    window = sg.Window('Torch Esgeta - Methods selector', font=("Helvetica", 12), resizable=True,
                       auto_size_text=True, auto_size_buttons=True).Layout(layout)

    button, values = window.Read()
    sg.SetOptions(text_justification='left')
    if disp:
        print(button, values)
    methods = [
        'ShapleyValueSampling_captum',
        'Feature_Ablation_captum',
        'Occlusion_captum',
        'score_cam_CNN Visualization',
        'guided_backprop_CNN Visualization',
        'Guided_Backprop_captum',
        'Guidedbackprop_torchray',
        'Saliency_captum',
        'input_x_gradient_captum',
        'integrated_grad_CNN Visualization',
        'Deconvolution_captum',
        'Deconv_torchray',
        'grad_times_image_CNN Visualization',
        'Deep_Lift_captum',
        'deep_lift_shap_captum',
        'Integrated_Gradients_captum',
        'gradient_shap_captum',
        'Guided_Grad_Cam_captum',
        'guided_grad_cam_CNN Visualization',
        'vanilla_backprop_CNN Visualization',
        'layer_visualization_CNN Visualization',
        'layer_conductance_captum',
        'Layer_Activation_captum',
        'internal_influence_captum',
        'layer_grad_shap_captum',
        'LayerDeepLift_captum',
        'layer_activation_guided_backprop_CNN Visualization',
        'LayerGradCam_captum',
        'grad_cam_CNN Visualization',
        'LayerGradientXActivation_captum',
        'Gradcam_torchray',
        'Gradient_torchray',
        'contrast_excitation_backprop_torchray',
        'contrast_excitation_backprop_torchray',
        'Linearapprox_torchray',
        'lucent_render_vis_lucent',
        'segmentation_lime',
        'DeepDream_CNN Visualization'
    ]
    method_names = []
    for i in values:
        if values[i]:
            method_names.append(methods[i])

    print(method_names)
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
    jsonFilePath = "TorchEsegeta_cfg_Misc_Seg_3d.json"
    main(jsonFilePath)
