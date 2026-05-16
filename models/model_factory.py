from models.SimpleCnn import SimpleCNN
from models.resnet18 import ResNet18
from models.resnet34 import ResNet34
from models.wideresnet import WideResNet
from models.vit import VisionTransformer


MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "wideresnet": WideResNet,
    "vit": VisionTransformer
}

def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unkown model: {model_name}")
    
    return MODEL_REGISTRY[model_name](**kwargs)
