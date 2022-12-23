import os
import torch
from models.cspconvnext import cspconvnext_t


if __name__ == '__main__':

    root = '../classification_pytorch/'
    model_dir = os.path.join(root, 'models_save/cspconvnext_t_165_0.71224')

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_model = cspconvnext_t(num_classes=257)
    param = torch.load(f'{model_dir}.pth')
    torch_model.load_state_dict(param, strict = False)


    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      f'{model_dir}.onnx',   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=13,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}
                                   }
                     )
    print('Onnx model has been exported!!!')