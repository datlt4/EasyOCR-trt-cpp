import os
import torch
import onnx
import argparse
from easyocr import *
from collections import OrderedDict
import types
import importlib


class Args():
    def __init__(self) -> None:
        # self.pth = "saved_model/craft_mlt_25k.pth"
        # self.onnx = "saved_model/detectionModel-cpu.onnx"
        # self.device = "cpu"
        # self.detection = True
        # self.recognition = False

        # self.pth = "saved_model/craft_mlt_25k.pth"
        # self.onnx = "saved_model/detectionModel-cuda.onnx"
        # self.device = "cpu"
        # self.detection = True
        # self.recognition = False

        self.pth = "saved_model/english_g2.pth"
        self.onnx = "saved_model/recognitionModel-cpu.onnx"
        self.device = "cpu"
        self.detection = False
        self.recognition = True

        # self.pth = "saved_model/english_g2.pth"
        # self.onnx = "saved_model/recognitionModel-cuda.onnx"
        # self.device = "cuda"
        # self.detection = False
        # self.recognition = True


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def validate(onnx_path, remove=True):
    if os.path.exists(onnx_path):
        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            print('[ ERROR ] The output onnx model is invalid: %s' % e)
            if remove:
                os.remove(onnx_path)
        else:
            print('[ PASSED ] The output onnx model is valid!')
    else:
        print('[ FAILED ] Failed when !')


if __name__ == "__main__":

    # args = Args()
    parser = argparse.ArgumentParser(description='Transform EasyOCR weights to ONNX.')
    parser.add_argument('--pth', "-p", type=str, help='weights file')
    parser.add_argument('--onnx', "-o", type=str, help='output onnx file')
    parser.add_argument("--device", "-g", type=str, help='cuda/cpu')
    parser.add_argument("--detection", "-d", action="store_true")
    parser.add_argument("--recognition", "-r", action="store_true")
    args = parser.parse_args()

    if args.detection:
        # CRAFT
        craft_net = CRAFT()

        if args.device == 'cpu':
            state_dict = torch.load(args.pth, map_location="cpu")
            new_state_dict = copyStateDict(state_dict)
            craft_net.load_state_dict(new_state_dict)
            try:
                torch.quantization.quantize_dynamic(
                    craft_net, dtype=torch.qint8, inplace=True)
            except:
                pass
        else:
            state_dict = torch.load(args.pth, map_location="cuda")
            new_state_dict = copyStateDict(state_dict)
            craft_net.load_state_dict(new_state_dict)
            craft_net = torch.nn.DataParallel(craft_net).to(args.device)
            craft_net = craft_net.to(args.device)
            torch.backends.cudnn.benchmark = False

        with torch.no_grad():
            craft_net.eval()

            batch_size_1 = 500
            batch_size_2 = 500
            in_shape = [1, 3, batch_size_1, batch_size_2]
            dummy_input = torch.rand(in_shape)
            dummy_input = dummy_input.to(args.device)

            torch.onnx.export(
                craft_net.module if args.device == "cuda" else craft_net,
                dummy_input,
                args.onnx,
                export_params=True,
                opset_version=11,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {2: 'batch_size_1', 3: 'batch_size_2'}},
            )
            validate(args.onnx)

    elif args.recognition:
        # Recognizer
        character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        separator_list = {}
        dict_list = {"en": "easyocr/dict/en.txt"}
        recognizer_converter = CTCLabelConverter(
            character, separator_list, dict_list)
        num_class = len(recognizer_converter.character)
        recog_network = 'generation2'

        recognizer_params = {"input_channel": 1,
                             "output_channel": 256, "hidden_size": 256}
        recognizer_model_pkg = importlib.import_module(
            "easyocr.model.vgg_model")
        recognizer_model = recognizer_model_pkg.Model(
            num_class=num_class, **recognizer_params)

        if args.device == 'cpu':
            state_dict = torch.load(args.pth, map_location=args.device)
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key[7:]
                new_state_dict[new_key] = value
            recognizer_model.load_state_dict(new_state_dict)
            try:
                torch.quantization.quantize_dynamic(
                    recognizer_model, dtype=torch.qint8, inplace=True)
            except:
                pass
        else:
            # Override the forward method to flatten parameters when in a multi-GPU environment
            def dp_forward(self, input):
                self.rnn.flatten_parameters()
                return self.forward_(input)

            for m in recognizer_model.modules():
                if type(m) is BidirectionalLSTM:
                    m.forward_ = m.forward
                    m.forward = types.MethodType(dp_forward, m)

            recognizer_model = torch.nn.DataParallel(
                recognizer_model).to(args.device)
            recognizer_model.load_state_dict(
                torch.load(args.pth, map_location=args.device))

        with torch.no_grad():
            recognizer_model.eval()

            batch_size_1_1 = 500
            in_shape_1 = [1, 1, 64, batch_size_1_1]
            dummy_input_1 = torch.rand(in_shape_1)
            dummy_input_1 = dummy_input_1.to(args.device)

            batch_size_2_1 = 50
            in_shape_2 = [1, batch_size_2_1]
            dummy_input_2 = torch.rand(in_shape_2)
            dummy_input_2 = dummy_input_2.to(args.device)

            dummy_input = (dummy_input_1, dummy_input_2)

            torch.onnx.export(
                recognizer_model.module if args.device == "cuda" else recognizer_model,
                dummy_input,
                args.onnx,
                export_params=True,
                opset_version=11,
                input_names=['input1', 'input2'],
                output_names=['output'],
                dynamic_axes={'input1': {3: 'batch_size_1_1'}},
            )

        validate(args.onnx)

# python -c "import easyocr; reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='./saved_model', download_enabled=False); print(reader.readtext('/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/EasyOCR/examples/english.png', detail=1))"
