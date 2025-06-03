import torch
from model_utils.mymodels import EfficientNetV2, ConvNeXt, ResNet50V2
from collections import OrderedDict


def load_model(model_name, path):
    import torch
    from collections import OrderedDict

    if model_name == "EfficientNetV2":
        model = EfficientNetV2(num_classes=12, hidden_units=342, dropout_rate=0.282)
    elif model_name == "ConvNeXt":
        model = ConvNeXt(num_classes=12, hidden_units=86, dropout_rate=0.273346)
    elif model_name == "ResNet50V2":
        model = ResNet50V2(num_classes=12, hidden_units=375, dropout_rate=0.181626)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    checkpoint = torch.load(path, map_location='cpu')

    # Extract state_dict from checkpoint if needed
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    if 'model' in checkpoint:
        print("Checkpoint keys and their shapes:")
        for k, v in checkpoint['model'].items():
            print(f"{k}: {tuple(v.shape)}")
        state_dict = checkpoint['model']
    else:
        print("Checkpoint keys and their shapes:")
        for k, v in checkpoint.items():
            print(f"{k}: {tuple(v.shape)}")
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        # Remove 'module.' prefix if exists
        if new_key.startswith('module.'):
            new_key = new_key[len('module.'):]

        # Remove 'model.' prefix if exists (your model keys start with 'model.')
        # But since your model keys start with 'model.', **keep it** to match your model keys
        # So do NOT remove 'model.' here, keep it.

        new_state_dict[new_key] = v

    # Now load state dict
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
