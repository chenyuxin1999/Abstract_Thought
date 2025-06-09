import json
from collections import defaultdict
import csv
import argparse


def get_layerwise_neuron_sets(*neuron_dicts):
    param_keys = neuron_dicts[0].keys()
    layer_sets = []

    for d in neuron_dicts:
        param_layer_set = defaultdict(dict)
        for param in param_keys:
            for i in range(len(d[param])):
                layer_set = set(d[param][str(i)])
                param_layer_set[param][i] = layer_set
        layer_sets.append(param_layer_set)
    
    return param_keys, layer_sets

def compute_intersection_and_exclusive(param_keys, layer_sets):
    num_layers = len(layer_sets[0][list(param_keys)[0]])

    shared_neurons = {param: {} for param in param_keys}
    exclusive_neurons = {lang_idx: {param: {} for param in param_keys} for lang_idx in range(len(layer_sets))}

    for param in param_keys:
        for layer in range(num_layers):
            sets_at_layer = [layer_sets[i][param][layer] for i in range(len(layer_sets))]
            # 交集
            shared = set.intersection(*sets_at_layer)
            shared_neurons[param][layer] = shared

            # 专属 neuron
            for idx, s in enumerate(sets_at_layer):
                exclusive = s - shared
                exclusive_neurons[idx][param][layer] = exclusive
    
    return shared_neurons, exclusive_neurons


def get_total_neurons_per_layer(model_name, param):
    model_name = model_name.lower()
    if "llama-3.2-1b" in model_name:
        return 8192 if "fwd" in param else 2048
    elif "llama-3.2-3b" in model_name:
        return 8192 if "fwd" in param else 3072
    elif "llama-3.1-8b" in model_name:
        return 14336 if "fwd" in param else 4096
    elif "gemma-3-4b" in model_name:
        return 10240 if "fwd" in param else 2048
    elif "gemma-2-9b" in model_name:
        return 14336 if "fwd" in param else 4096
    elif "qwen2.5-7b" in model_name:
        return 18944 if "fwd" in param else 3584
    elif "qwen2.5-3b" in model_name:
        return 11008 if "fwd" in param else 2048
    elif "qwen2.5-0.5b" in model_name:
        return 4846 if "fwd" in param else 896
    elif "qwen2.5-1.5b" in model_name:
        return 8960 if "fwd" in param else 1536
    elif "qwen2-7b" in model_name:
        return 18944 if "fwd" in param else 3584
    elif "qwen1.5-7b" in model_name:
        return 11008 if "fwd" in param else 4096
    elif "gemma-7b" in model_name:
        return 24576 if "fwd" in param else 4096
    elif "llama3-8b" in model_name:
        return 14336 if "fwd" in param else 4096
    elif "llama-2-7b" in model_name:
        return 11008 if "fwd" in param else 4096
    elif "llama-1-7b" in model_name:
        return 11008 if "fwd" in param else 4096
    elif "llama-3-8b" in model_name:
        return 14336 if "fwd" in param else 4096
    elif "qwen2-0.5b" in model_name:
        return 4864 if "fwd" in param else 896
    elif "qwen2-1.5b" in model_name:
        return 8960 if "fwd" in param else 1536
    elif "qwen1.5-0.5b" in model_name:
        return 2016 if "fwd" in param else 1024
    elif "qwen1.5-1.8b" in model_name:
        return 5504 if "fwd" in param else 2048
    elif "qwen1.5-4b" in model_name:
        return 6912 if "fwd" in param else 2560
    elif "qwen3-0.6b" in model_name:
        return 3072 if "fwd" in param else 2048
    elif "qwen3-1.7b" in model_name:
        return 6144 if "fwd" in param else 2048
    else:
        raise ValueError(f"Unknown model type in: {model_name}")
        
def print_shared_exclusive_percentages(shared_neurons, exclusive_neurons, model_name):
    # print(f"\n==== Analyzing neuron percentages for model: {model_name} ====")
    lang_dict = {0: "en", 1: "zh", 2: "th", 3: "fr", 4: "de", 5: "sw"}
    # lang_dict = {0: "zh", 1: "th", 2: "fr", 3: "de", 4: "sw"}
    num_languages = len(exclusive_neurons)
    
    grand_total_neurons = 0
    grand_total_shared = 0
    grand_total_exclusive = [0] * num_languages

    for param in shared_neurons:
        # print(f"\n-- Parameter: {param} --")
        param_total = 0
        param_shared = 0
        param_exclusive = [0] * num_languages
        for layer in shared_neurons[param]:
            layer_total = get_total_neurons_per_layer(model_name, param)
            param_total += layer_total
            grand_total_neurons += layer_total

            shared_count = len(shared_neurons[param][layer])
            param_shared += shared_count
            grand_total_shared += shared_count

            for i in range(num_languages):
                excl_count = len(exclusive_neurons[i][param][layer])
                param_exclusive[i] += excl_count
                grand_total_exclusive[i] += excl_count
        
        # print(f"Total % of shared neurons in {param}: {param_shared / param_total * 100:.2f}%")
        # for i in range(num_languages):
            # print(f"Total % of exclusive neurons in {param} for lang-{lang_dict[i]}: {param_exclusive[i] / param_total * 100:.2f}%")
    


    # print(f"\n==== Overall summary ====")
    # print(f"Shared neurons %: {grand_total_shared / grand_total_neurons * 100:.2f}%")
    # for i in range(num_languages):
    #     print(f"Exclusive neurons % for lang-{lang_dict[i]}: {grand_total_exclusive[i] / grand_total_neurons * 100:.2f}%")
    return grand_total_shared,  grand_total_exclusive


def print_original_percentages(all_lang_data, model_name):
    # print(f"\n==== Original neuron percentages per language for model: {model_name} ====")
    lang_dict = {0: "en", 1: "zh", 2: "th", 3: "fr", 4: "de", 5: "sw"}
    num_languages = len(all_lang_data)

    grand_total_neurons = 0
    grand_total_selected = [0] * num_languages

    param_keys = all_lang_data[0].keys()
    num_layers = len(all_lang_data[0][list(param_keys)[0]])

    for param in param_keys:
        # print(f"\n-- Parameter: {param} --")
        param_total = 0
        param_selected = [0] * num_languages
        for layer in range(num_layers):
            layer_total = get_total_neurons_per_layer(model_name, param)
            param_total += layer_total
            grand_total_neurons += layer_total

            for i in range(num_languages):
                count = len(all_lang_data[i][param][str(layer)])
                param_selected[i] += count
                grand_total_selected[i] += count
        
    #     for i in range(num_languages):
    #         print(f"Original % of neurons in {param} for lang-{lang_dict[i]}: {param_selected[i] / param_total * 100:.2f}%")

    # print(f"\n==== Overall original summary ====")
    # for i in range(num_languages):
    #     print(f"Original neurons % for lang-{lang_dict[i]}: {grand_total_selected[i] / grand_total_neurons * 100:.2f}%")

    return grand_total_selected, grand_total_neurons


def save_exclusive_neurons_as_json(exclusive_neurons, lang, output_path):
    lang_dict = {"en": 0, "zh": 1, "th": 2, "fr": 3, "de": 4, "sw": 5}
    # lang_dict = {"zh": 0, "th": 1, "fr": 2, "de": 3, "sw": 4}
    lang_idx = lang_dict[lang]
    # lang_idx = 0 for lang-en
    exclusive_lang = exclusive_neurons[lang_idx]
    
    formatted = {}
    for param in exclusive_lang:
        formatted[param] = {}
        for layer in exclusive_lang[param]:
            formatted[param][str(layer)] = list(exclusive_lang[param][layer])  # set -> list
    
    # 保存为 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(formatted, f, indent=2)
    
    print(f"Saved exclusive neurons for lang-{lang} to {output_path}")


def save_shared_neurons_to_json(shared_neurons, output_path):
    formatted = {}

    for param in shared_neurons:
        formatted[param] = {}
        for layer in shared_neurons[param]:
            formatted[param][str(layer)] = list(shared_neurons[param][layer])  # ensure str keys + list values

    with open(output_path, 'w') as f:
        json.dump(formatted, f, indent=2)

    print(f"✅ Shared neurons saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B", help="Model name, e.g., Qwen2.5-7B")
    parser.add_argument("--ratio", type=str, default="0.01", help="Neuron selection ratio")
    parser.add_argument("--neurons_path", type=str, default="./neuron_train_data_detect", help="Path to neuron JSON files")
    parser.add_argument("--save_path", type=str, default="./neuron_deactivation", help="Output path for shared and exclusive neuron files")
    parser.add_argument("--record_path", type=str, default="./neurons_statistics.csv", help="CSV file to save neuron statistics")

    args = parser.parse_args()

    model_name = args.model_name
    ratio = args.ratio
    neurons_path = args.neurons_path
    save_path = args.save_path
    record_path = args.record_path

    all_lang_neurons = defaultdict(list)
    for lang in ["en", "zh", "th", "sw", "fr", "de"]:
        with open(f"{neurons_path}/{model_name}_gsm_{lang}_atten{ratio}_ffn{ratio}.json", "r") as f:
            neurons = json.load(f)
            all_lang_neurons[lang] = neurons


    param_keys, layer_sets = get_layerwise_neuron_sets(
        all_lang_neurons['en'], all_lang_neurons['zh'], all_lang_neurons['th'],
        all_lang_neurons['fr'], all_lang_neurons['de'], all_lang_neurons['sw']
    )
    shared_neurons, exclusive_neurons = compute_intersection_and_exclusive(param_keys, layer_sets)
    shared_neurons_num, exclusive_neurons_num_list = print_shared_exclusive_percentages(
        shared_neurons=shared_neurons,
        exclusive_neurons=exclusive_neurons,
        model_name=model_name
    )


    all_lang_data = [
        all_lang_neurons['en'],
        all_lang_neurons['zh'],
        all_lang_neurons['th'],
        all_lang_neurons['fr'],
        all_lang_neurons['de'],
        all_lang_neurons['sw']
    ]
    original_neurons_num, total_neurons_num = print_original_percentages(all_lang_data, model_name=model_name)

    lang_dict = {0: "en", 1: "zh", 2: "th", 3: "fr", 4: "de", 5: "sw"}

    print(f"Model: {model_name}, Ratio: {ratio}")
    print(f"Total neurons: {total_neurons_num}")
    for i in range(len(original_neurons_num)):
        print(f"Active neurons for lang-{lang_dict[i]}: {original_neurons_num[i]}, {original_neurons_num[i] / total_neurons_num * 100}%")

    print(f"Shared neurons: {shared_neurons_num}, {shared_neurons_num / total_neurons_num * 100}%")
    for i in range(len(exclusive_neurons_num_list)):
        print(f"Exclusive neurons for lang-{lang_dict[i]}: {exclusive_neurons_num_list[i]}, {exclusive_neurons_num_list[i] / total_neurons_num * 100}%")

    average_active_neurons_num = sum(original_neurons_num) / len(original_neurons_num)
    average_share_neurons_num = shared_neurons_num / average_active_neurons_num * 100
    print(f"Average share neurons Percentage v.s. active neurons: {average_share_neurons_num}%")

    with open(record_path, mode="a", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        if f_out.tell() == 0:
            writer.writerow(["model_name", "ratio", "shared_neuron", "en", "zh", "th", "fr", "de", "sw", "total_neurons_num", "shared_neuron_num", "en_num", "zh_num", "th_num", "fr_num", "de_num", "sw_num", "relative_shared_neuron"])

        writer.writerow([model_name, ratio, f"{round(shared_neurons_num/total_neurons_num*100, 4)}%", f"{round(exclusive_neurons_num_list[0]/total_neurons_num*100, 4)}%", f"{round(exclusive_neurons_num_list[1]/total_neurons_num*100, 4)}%", f"{round(exclusive_neurons_num_list[2]/total_neurons_num*100, 4)}%", f"{round(exclusive_neurons_num_list[3]/total_neurons_num*100, 4)}%", f"{round(exclusive_neurons_num_list[4]/total_neurons_num*100, 4)}%", f"{round(exclusive_neurons_num_list[5]/total_neurons_num*100, 4)}%", total_neurons_num, shared_neurons_num, exclusive_neurons_num_list[0], exclusive_neurons_num_list[1], exclusive_neurons_num_list[2], exclusive_neurons_num_list[3], exclusive_neurons_num_list[4], exclusive_neurons_num_list[5], f"{round(average_share_neurons_num, 4)}%"])
        

    save_shared_neurons_to_json(shared_neurons, f"{save_path}/{model_name}_gsm_shared_neurons_{ratio}.json")

    save_exclusive_neurons_as_json(exclusive_neurons, lang="en", output_path=f"{save_path}/{model_name}_gsm_exclusive_en_neuron_{ratio}.json")
    save_exclusive_neurons_as_json(exclusive_neurons, lang="zh", output_path=f"{save_path}/{model_name}_gsm_exclusive_zh_neuron_{ratio}.json")
    save_exclusive_neurons_as_json(exclusive_neurons, lang="th", output_path=f"{save_path}/{model_name}_gsm_exclusive_th_neuron_{ratio}.json")
    save_exclusive_neurons_as_json(exclusive_neurons, lang="fr", output_path=f"{save_path}/{model_name}_gsm_exclusive_fr_neuron_{ratio}.json")
    save_exclusive_neurons_as_json(exclusive_neurons, lang="de", output_path=f"{save_path}/{model_name}_gsm_exclusive_de_neuron_{ratio}.json")
    save_exclusive_neurons_as_json(exclusive_neurons, lang="sw", output_path=f"{save_path}/{model_name}_gsm_exclusive_sw_neuron_{ratio}.json")
    print(f"Saved neurons to {save_path}")





