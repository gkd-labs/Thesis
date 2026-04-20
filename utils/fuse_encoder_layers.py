import torch
import os
import sys

from model.chemgraphormer import ChemGraphormerClassifier   # make sure model folder is present in the current working directory

def fuse_encoders(checkpoint1_path, checkpoint2_path, mae1, mae2, output_path):
    """
    Loads two full model checkpoints, extracts their encoders,
    and fuses them into a single encoder checkpoint using a weighted average.
    """
    # Load state dictionaries of the two full best model model checkpoints
    state_dict1 = torch.load(checkpoint1_path, map_location='cpu', weights_only=True)
    state_dict2 = torch.load(checkpoint2_path, map_location='cpu', weights_only=True)

    # Calculate normalized weights based on inverse MAE and apply them to the individual model states before fusion
    weight1 = 1 / mae1
    weight2 = 1 / mae2
    total_weight = weight1 + weight2
    normalized_weight1 = weight1 / total_weight
    normalized_weight2 = weight2 / total_weight

    fused_chemgraphormer_encoder_state_dict = {}

    for key in state_dict1.keys():
           if key.startswith('encoder.'):
            encoder_key = key.replace('encoder.', '')
            param1 = state_dict1[key]
            param2 = state_dict2[key]
            fused_param = (param1 * normalized_weight1) + (param2 * normalized_weight2)
            fused_chemgraphormer_encoder_state_dict[encoder_key] = fused_param

    # Save the fused state dictionary
    torch.save(fused_chemgraphormer_encoder_state_dict, output_path)
    print(f"Fused encoder saved to: {output_path}")