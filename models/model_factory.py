from models.SimpleCnn import SimpleCNN
from models.resnet18 import ResNet18
from models.resnet34 import ResNet34
from models.wideresnet import WideResNet

MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "wideresnet": WideResNet,
}

def get_model(model_name, **kwargs):
    if model_name in MODEL_REGISTRY:
        raise ValueError(f"Unkown model: {model_name}")
    return MODEL_REGISTRY(model_name)(**kwargs)
