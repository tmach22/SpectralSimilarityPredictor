import torch
import torch.nn as nn
import yaml
import os
import sys
from pathlib import Path

# --- SETUP ---
cwd = Path.cwd()
sys.path.insert(0, os.path.join(cwd, 'model', 'transport_model'))
sys.path.insert(0, os.path.join(os.path.dirname(cwd.parent), 'tmach007/massformer/src/massformer'))

# Import your unpooled encoder
from classifier_siamese_model import UnpooledMassFormerEncoder

def load_and_merge_configs(template_path="/data/nas-gpu/wang/tmach007/massformer/config/template.yml", custom_path="/data/nas-gpu/wang/tmach007/massformer/config/demo/demo_eval.yml"):
    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open(custom_path, 'r', encoding='utf-8') as f:
        custom_config = yaml.safe_load(f)
        
    for section, subdict in custom_config.items():
        if isinstance(subdict, dict):
            if section not in config:
                config[section] = {}
            for k, v in subdict.items():
                config[section][k] = v
        else:
            config[section] = subdict
    return config

def inspect_massformer():
    print("Loading merged configs (Template + Demo)...")
    full_config = load_and_merge_configs()
    
    print("Initializing UnpooledMassFormerEncoder...")
    encoder = UnpooledMassFormerEncoder(full_config['model'], checkpoint_path=None)
    
    print("\n" + "="*50)
    print("ALL LINEAR LAYERS IN THE MASSFORMER ENCODER")
    print("="*50)
    
    linear_layer_names = []
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Linear):
            print(name)
            short_name = name.split('.')[-1]
            if short_name not in linear_layer_names:
                linear_layer_names.append(short_name)

    print("\n" + "="*50)
    print("UNIQUE SUFFIXES FOR LORA TARGET_MODULES:")
    print("="*50)
    print(linear_layer_names)

if __name__ == "__main__":
    inspect_massformer()