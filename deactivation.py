import torch
import numpy as np
from transformers import LlamaForCausalLM, LlamaConfig, Gemma2ForCausalLM, AutoConfig, AutoModelForCausalLM, Gemma3ForConditionalGeneration
from typing import Dict, List
import json
from collections import defaultdict
import argparse
import random
import math

def zero_out_indices(weight_tensor, dim, indices_to_zero, factor):
    weight_tensor = weight_tensor.clone()
    indices_to_zero = [i // factor for i in indices_to_zero]
    if dim == 0:
        weight_tensor[indices_to_zero, :] = 0
    elif dim == 1:
        weight_tensor[:, indices_to_zero] = 0
    return weight_tensor

def apply_zero_mask_to_model(model, deactivate_indices_dict):
    state_dict = model.state_dict()
    new_state_dict = {}
    for name, param in state_dict.items():
        if name in deactivate_indices_dict:
            info = deactivate_indices_dict[name]
            # print(f"Zeroing {name} at dim={info['dim']} for {len(info['indices'])} neurons")
            param = zero_out_indices(param, info["dim"], info["indices"], info["factor"])
        new_state_dict[name] = param
    model.load_state_dict(new_state_dict)

def build_deactivate_indices_dict(
    model_path: str,
    activate_keys_q_set: Dict[int, List[int]],
    activate_keys_k_set: Dict[int, List[int]],
    activate_keys_v_set: Dict[int, List[int]],
    activate_keys_o_set: Dict[int, List[int]],
    activate_keys_fwd_up_set: Dict[int, List[int]],
    activate_keys_fwd_down_set: Dict[int, List[int]],
    total_layers: int,
    intermediate_size: int,
    hidden_size: int,
):
    deactivate_dict = {}
    model_config = AutoConfig.from_pretrained(model_path)
    if hasattr(model_config, "text_config"):
        model_config = model_config.text_config
    kv_factor = model_config.num_attention_heads / model_config.num_key_value_heads
    print(f"kv_factor: {kv_factor}")

    for idx in range(total_layers):
        layer_idx = str(idx)

        if layer_idx in activate_keys_q_set:
            deactivate_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = {
                "dim": 0,
                "indices": list(set(activate_keys_q_set[layer_idx])),
                "factor": 1
            }
        if layer_idx in activate_keys_k_set:
            deactivate_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = {
                "dim": 0,
                "indices": list(set(activate_keys_k_set[layer_idx])),
                "factor": kv_factor
            }
        if layer_idx in activate_keys_v_set:
            deactivate_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = {
                "dim": 0,
                "indices": list(set(activate_keys_v_set[layer_idx])),
                "factor": kv_factor
            }
        if layer_idx in activate_keys_o_set:
            deactivate_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = {
                "dim": 1,
                "indices": list(set(activate_keys_o_set[layer_idx])),
                "factor": 1
            }

        # FFN Up / Down
        if layer_idx in activate_keys_fwd_up_set:
            deactivate_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = {
                "dim": 0,
                "indices": list(set(activate_keys_fwd_up_set[layer_idx])),
                "factor": 1
            }
        if layer_idx in activate_keys_fwd_down_set:
            deactivate_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = {
                "dim": 1,
                "indices": list(set(activate_keys_fwd_down_set[layer_idx])),
                "factor": 1
            }

    return deactivate_dict


import shutil
import os

def deactivation_params_saving(model_path, detected_neuron, save_path):
    # === Step 1: Load original model ===
    if "gemma-3-4b-pt" in model_path.lower():
        model = Gemma3ForConditionalGeneration.from_pretrained(model_path)
        model = model.language_model
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    config = model.config

    # === Step 2: Load active index sets ===
    # Example:
    # activate_keys_q_set = {0: [1, 2, 3], 1: [4, 5, 6]}
    # activate_keys_k_set = {0: [1, 3, 5], 1: [2, 4, 6]}
    # activate_keys_v_set = {0: [0, 1], 1: [2, 3]}
    # activate_keys_fwd_up_set = {0: [100, 200], 1: [150, 250]}
    # activate_keys_fwd_down_set = {0: [50, 60], 1: [70, 80]}
    activate_keys_q_set = detected_neuron['attn_q']
    activate_keys_k_set = detected_neuron['attn_k']
    activate_keys_v_set = detected_neuron['attn_v']
    activate_keys_o_set = detected_neuron['attn_o']
    activate_keys_fwd_up_set = detected_neuron['fwd_up']
    activate_keys_fwd_down_set = detected_neuron['fwd_down']

    # === Step 3: Build deactivate dict ===
    deactivate_dict = build_deactivate_indices_dict(
        model_path,
        activate_keys_q_set,
        activate_keys_k_set,
        activate_keys_v_set,
        activate_keys_o_set,
        activate_keys_fwd_up_set,
        activate_keys_fwd_down_set,
        total_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        hidden_size=config.hidden_size
    )

    # === Step 4: Zero out weights ===
    apply_zero_mask_to_model(model, deactivate_dict)

    # === Step 5: Save new model ===
    model.save_pretrained(save_path)
    print(f"✅ Modified model saved to: {save_path}")

    # === Step 6: Copy tokenizer files ===
    tokenizer_source_path = model_path
    for fname in os.listdir(tokenizer_source_path):
        if "tokenizer" in fname:
            src = os.path.join(tokenizer_source_path, fname)
            dst = os.path.join(save_path, fname)
            shutil.copy2(src, dst)
    print(f"📝 Tokenizer files copied from {tokenizer_source_path} to {save_path}")


def print_original_percentages(all_lang_data, model_name):
    # print(f"\n==== Original neuron percentages per language for model: {model_name} ====")
    num_languages = len(all_lang_data)

    grand_total_selected = [0] * num_languages

    param_keys = all_lang_data[0].keys()
    num_layers = len(all_lang_data[0][list(param_keys)[0]])

    for param in param_keys:
        # print(f"\n-- Parameter: {param} --")
        param_total = 0
        param_selected = [0] * num_languages
        for layer in range(num_layers):
            for i in range(num_languages):
                count = len(all_lang_data[i][param][str(layer)])
                param_selected[i] += count
                grand_total_selected[i] += count
        
    #     for i in range(num_languages):
    #         print(f"Original % of neurons in {param} for lang-{lang_dict[i]}: {param_selected[i] / param_total * 100:.2f}%")

    # print(f"\n==== Overall original summary ====")
    # for i in range(num_languages):
    #     print(f"Original neurons % for lang-{lang_dict[i]}: {grand_total_selected[i] / grand_total_neurons * 100:.2f}%")

    return grand_total_selected

def generate_random_neuron_uniform_distribution(data, file_name, seed=42):
    random.seed(seed)
    random_data = {}


    total_neuron_num = 0
    total_layer_count = 0
    for param in data:
        for layer in data[param]:
            total_neuron_num += len(data[param][layer])
            total_layer_count += 1

    avg_neurons_per_layer = math.ceil(total_neuron_num / total_layer_count)

    print(f"Target total neurons: {total_neuron_num}")
    print(f"Avg neurons per layer: {avg_neurons_per_layer}")


    param_neuron_counter = {}
    actual_total_neuron = 0

    for param in data:
        param_neuron_counter[param] = 0
        random_data[param] = {}

        for layer in data[param]:
            if "llama-3.2-1b" in file_name.lower():
                total_neurons = 8192 if "fwd" in param else 2048
            elif "llama-3.2-3b" in file_name.lower():
                total_neurons = 8192 if "fwd" in param else 3072
            elif "llama-3.1-8b" in file_name.lower():
                total_neurons = 14336 if "fwd" in param else 4096
            elif "gemma-3-4b" in file_name.lower():
                total_neurons = 10240 if "fwd" in param else 2048
            elif "gemma-2-9b" in file_name.lower():
                total_neurons = 14336 if "fwd" in param else 4096
            elif "qwen2.5-7b" in file_name.lower():
                total_neurons = 18944 if "fwd" in param else 3584
            elif "qwen2.5-3b" in file_name.lower():
                total_neurons = 11008 if "fwd" in param else 2048
            elif "qwen2.5-0.5b" in file_name.lower():
                total_neurons = 4846 if "fwd" in param else 896
            elif "qwen2.5-1.5b" in file_name.lower():
                total_neurons = 8960 if "fwd" in param else 1536
            elif "qwen1.5-7b" in file_name.lower():
                total_neurons = 11008 if "fwd" in param else 4096
            elif "qwen2-7b" in file_name.lower():
                total_neurons = 18944 if "fwd" in param else 3584
            elif "gemma-7b" in file_name.lower():
                total_neurons = 24576 if "fwd" in param else 4096
            elif "gemma-3-4b-pt" in file_name.lower():
                total_neurons = 10240 if "fwd" in param else 2048
            elif "llama3-8b" in file_name.lower():
                total_neurons = 14336 if "fwd" in param else 4096
            elif "llama-2-7b" in file_name.lower():
                total_neurons = 11008 if "fwd" in param else 4096
            elif "llama-1-7b" in file_name.lower():
                total_neurons = 11008 if "fwd" in param else 4096
            elif "llama-3-8b" in file_name.lower():
                total_neurons = 14336 if "fwd" in param else 4096
            elif "qwen2-0.5b" in file_name.lower():
                total_neurons = 4864 if "fwd" in param else 896
            elif "qwen2-1.5b" in file_name.lower():
                total_neurons = 8960 if "fwd" in param else 1536
            elif "qwen1.5-0.5b" in file_name.lower():
                total_neurons = 2016 if "fwd" in param else 1024
            elif "qwen1.5-1.8b" in file_name.lower():
                total_neurons = 5504 if "fwd" in param else 2048
            elif "qwen1.5-4b" in file_name.lower():
                total_neurons = 6912 if "fwd" in param else 2560
            elif "qwen3-0.6b" in file_name.lower():
                total_neurons = 3072 if "fwd" in param else 2048
            elif "qwen3-1.7b" in file_name.lower():
                total_neurons = 6144 if "fwd" in param else 2048
            else:
                raise ValueError("Unknown model type in file name.")

            num_sample = min(avg_neurons_per_layer, total_neurons)
            sampled = random.sample(range(total_neurons), num_sample)
            random_data[param][str(layer)] = sampled

            param_neuron_counter[param] += num_sample
            actual_total_neuron += num_sample


    print("\n--- Neuron Distribution by Parameter ---")
    for param, count in param_neuron_counter.items():
        percentage = count / actual_total_neuron * 100
        print(f"{param}: {count} neurons ({percentage:.6f}%)")

    print(f"\nTotal generated neurons: {actual_total_neuron} (original: {total_neuron_num})")

    return random_data


def save_neurons_to_json(neurons, output_path):
    formatted = {}

    for param in neurons:
        formatted[param] = {}
        for layer in neurons[param]:
            formatted[param][str(layer)] = list(neurons[param][layer])  # ensure str keys + list values

    with open(output_path, 'w') as f:
        json.dump(formatted, f, indent=2)

    print(f"✅ Neurons saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B", help="Model name, e.g., Qwen2.5-7B")
    parser.add_argument("--ratio", type=str, default="0.01", help="Neuron selection ratio")
    parser.add_argument("--neurons_path", type=str, default="./neuron_deactivation", help="Path to neuron JSON files")
    parser.add_argument("--save_path", type=str, default="./deactivate_model_param", help="Output path for model files")
    parser.add_argument("--model_path", type=str, default="", help="Model path")

    args = parser.parse_args()

    model_name = args.model_name
    ratio = args.ratio
    neurons_path = args.neurons_path
    save_path = args.save_path
    model_path = args.model_path

    all_exclusive_lang_neurons = defaultdict(list)
    for lang in ["en", "zh", "th", "sw", "fr", "de"]:
        with open(f"{neurons_path}/{model_name}_gsm_exclusive_{lang}_neuron_{ratio}.json", "r") as f:
            neurons = json.load(f)
            all_exclusive_lang_neurons[lang] = neurons

    with open(f"{neurons_path}/{model_name}_gsm_shared_neurons_{ratio}.json", "r") as f:
        shared_neurons = json.load(f)

    all_lang_neurons = [
        all_exclusive_lang_neurons['en'],
        all_exclusive_lang_neurons['zh'],
        all_exclusive_lang_neurons['th'],
        all_exclusive_lang_neurons['sw'],
        all_exclusive_lang_neurons['fr'],
        all_exclusive_lang_neurons['de']
    ]

    neurons_num = print_original_percentages(all_lang_neurons, model_name)

    # find the lang with the most neurons
    # max_neurons_lang_idx = np.argmax(neurons_num)
    # lang_dict = {0: "en", 1: "zh", 2: "th", 3: "sw", 4: "fr", 5: "de"}
    # max_neurons_lang = lang_dict[max_neurons_lang_idx]
    # print(f"The language with the most neurons is: {max_neurons_lang}")
    

    # most_exclusive_lang_neurons = all_exclusive_lang_neurons[max_neurons_lang]

    random_neuron = generate_random_neuron_uniform_distribution(shared_neurons, model_name)
    save_neurons_to_json(random_neuron, f"./neuron_deactivation/{model_name}_gsm_random_neurons_{ratio}.json")


    deactivation_params_saving(model_path=f"{model_path}/{model_name}", detected_neuron=shared_neurons, save_path=f"{save_path}/{model_name}_shared_neuron_{ratio}")
    # deactivation_params_saving(model_path=f"{model_path}/{model_name}", detected_neuron=most_exclusive_lang_neurons, save_path=f"{save_path}/{model_name}_most_exclusive_lang_neuron_{ratio}")
    deactivation_params_saving(model_path=f"{model_path}/{model_name}", detected_neuron=random_neuron, save_path=f"{save_path}/{model_name}_random_neuron_{ratio}")


    # save models deactivated by all exclusive neurons
    for lang in ["en", "zh", "th", "sw", "fr", "de"]:
        deactivation_params_saving(model_path=f"{model_path}/{model_name}", detected_neuron=all_exclusive_lang_neurons[lang], save_path=f"{save_path}/{model_name}_{lang}_exclusive_lang_neuron_{ratio}")

    

    

    
    
