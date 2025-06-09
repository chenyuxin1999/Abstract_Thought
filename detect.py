import random, os, json, sys
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers_custom.modeling_gemma3 import Gemma3ForConditionalGeneration
from transformers_custom.modeling_llama import LlamaForCausalLMDetect
from transformers_custom.modeling_gemma2 import Gemma2ForCausalLM
from transformers_custom.modeling_qwen2 import Qwen2ForCausalLM
from transformers_custom.modeling_qwen3 import Qwen3ForCausalLM
from transformers_custom.modeling_gemma import GemmaForCausalLM
import argparse
from transformers import AutoConfig
import pandas as pd

def save_neuron(activate_neurons, path):
    for group in activate_neurons:
        entry = activate_neurons[group]
        activate_neurons[group] = {key: list(value) if isinstance(value, set) else value for key, value in entry.items()}
    with open(path, 'w') as f:
        json.dump(activate_neurons, f)

def load_lines_from_dataset(task, lang, args):
    if task == "detect":
        file_path = os.path.join(args.corpus_path, f"/detect/{lang}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Detect data file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    elif task == "gsm":
        suffix_map = {"zh": "_zh", "en": "_en", "fr": "_fr", "sw": "_sw", "de": "_de", "th": "_th"}
        file_path = f"{args.corpus_path}/test/mgsm/mgsm{suffix_map[lang]}.tsv"
        df = pd.read_csv(file_path, sep='\t', names=["question", "answer"])
        return [f"Question: {q}\nAnswer: {a}" for q, a in zip(df["question"], df["answer"])]

    elif task == "mmlu":
        suffix_map = {
            "en": "EN-EN", "zh": "ZH-CN", "fr": "FR-FR",
            "th": "TH-TH", "de": "DE-DE", "sw": "SW-KE"
        }
        file_path = f"{args.corpus_path}/test/mmlu/mmlu_{suffix_map[lang]}.csv"
        df = pd.read_csv(file_path)
        return [f"Question: {row['Question']}" for _, row in df.iterrows()]

    elif task == "m3exam":
        suffix_map = {
            "en": "english", "zh": "chinese", "th": "thai", "sw": "swahili"
        }
        test_path = f"{args.corpus_path}/test/m3exam/{suffix_map[lang]}-questions-test.json"
        questions = json.load(open(test_path))
        return [q["question_text"] for q in questions]

    else:
        raise ValueError(f"Unsupported data type: {task}")

def detect_key_neurons(model, tokenizer, lang,
                       atten_ratio=0.1, ffn_ratio=0.1, test_size=-1, candidate_layers=[], 
                       detection_path="./test_data/oscar", output_path="./output",
                       suffix = "", model_name = "", sample_size=10000, task="detect", args=None) -> dict:
    """Detects neurons key to the language *lang* and writes to ../output/model_lang_neuron.txt 

    Args:
        model (AutoModelForCausalLM): loaded hf model.
        tokenizer (AutoTokenizer): loaded hf tokenizer.
        lang (str): one of ['english', 'chinese', 'french', 'russian']
        test_size (int, optional): number of entries used when detecting.
        candidate_layers (list, optional): list of layers to examine.
    """
    if not len(candidate_layers):
        candidate_layers = range(model.config.num_hidden_layers)
    
    # with open(detection_path + lang + '.txt', 'r') as file:
    #     lines = file.readlines()
    # lines = [line.strip() for line in lines]
    lines = load_lines_from_dataset(task, lang, args)
    if sample_size > 0:
        if sample_size < len(lines):
            lines = random.sample(lines, sample_size)
    #lines = lines[:test_size] #Because using the same corpus for detection and training now, separating them.

    activate_key_sets = {
        "fwd_up" : [],
        "fwd_down" : [],
        "attn_q" : [],
        "attn_k" : [],
        "attn_v" : [],
        "attn_o" : []
    }
    error_count = 0

    intermediate_layers_decode = {}

    print("Detection corpus size: ", len(lines))
    count = 0
    for prompt in tqdm(lines):
        #hidden, answer, activate, o_layers = detection_prompting(model, tokenizer, prompt, candidate_layers)
        try:
            hidden, answer, activate, o_layers = detection_prompting(model, tokenizer, prompt, 
                                                                    candidate_layers, atten_ratio=atten_ratio, ffn_ratio=ffn_ratio)
            for key in activate.keys():
                activate_key_sets[key].append(activate[key])
            count += 1
            intermediate_layers_decode[count] = hidden
        except Exception as e:
            error_count += 1
            count += 1
            # Handle the OutOfMemoryError here
            print(error_count)
            print(e)

    print("Detection query complete; error: ", error_count)
    
    for group in activate_key_sets.keys():
        entries = activate_key_sets[group]
        common_layers = {}
        for layer in entries[0].keys():
            if all(layer in d for d in entries):
                arrays = [d[layer] for d in entries]
                common_elements = set.intersection(*map(set, arrays))
                common_elements = {int(x) for x in common_elements}

                common_layers[layer] = common_elements
        activate_key_sets[group] = common_layers
        print(f"{group} integrated and logged")
        '''
        DETECTION SET TO LAYER

        from collections import Counter
        threshold = test_size // 100
        
        common_layers = {}
        for layer in entries[0].keys():
            neuron_counter = Counter(neuron for d in entries for neuron in d[layer][0])
            frequent_neurons = [neuron for neuron, count in neuron_counter.items() if count > threshold]
            common_layers[layer] = set([int(x) for x in frequent_neurons])

        activate_key_sets[group] = common_layers'''
        
        
        #final structure of important neurons: {"param_set": {"layer1": [neuron1, neuron2, ...], ...}, ...}
    # file_name = f"{model.name_or_path.split('/')[-1]}_{lang}_atten{atten_ratio}_ffn{ffn_ratio}.json"
    # file_path = output_path + file_name
    # save_neuron(activate_key_sets, file_path)
    # save intermediate_layers_decode

    if "huggingface" in model_name:
        train_on_lang = "base_model"
        file_name_prefix = model_name.split('/')[-1]
    elif "llama" or "gemma" in model_name.split('/')[-2].lower():
        train_on_lang = model_name.split('/')[-2][:2] 
        file_name_prefix = model_name.split('/')[-2]
    elif "llama" or "gemma" in model_name.split('/')[-1].lower():
        train_on_lang = model_name.split('/')[-1][:2]
        file_name_prefix = model_name.split('/')[-1]
    else:
        raise ValueError(f"Model {model_name} not supported")

    # with open('./intermediate_decode_train_data_detect/' + f"{file_name_prefix}_{train_on_lang}_{lang}_atten{atten_ratio}_ffn{ffn_ratio}.json", 'w') as f:
    #     json.dump(intermediate_layers_decode, f)

    file_name = f"{file_name_prefix}_{task}_{lang}_atten{atten_ratio}_ffn{ffn_ratio}.json"
    file_path = output_path + file_name
    save_neuron(activate_key_sets, file_path)

        
    return activate_key_sets
        

def detection_prompting(model, tokenizer, prompt, candidate_premature_layers, atten_ratio=0.1, ffn_ratio=0.1):
    cut_off_len = 256
    inputs = tokenizer(prompt, return_tensors="pt")
    hidden_states, outputs, activate, o_layers = model.generate(**{'input_ids':inputs.input_ids[:, :cut_off_len].to(model.device),
                                                                   'attention_mask':inputs.attention_mask[:, :cut_off_len].to(model.device), 
                                                                   'max_new_tokens':1, 
                                                                   'candidate_premature_layers':candidate_premature_layers,
                                                                   'top_ratio_atten': atten_ratio,
                                                                   'top_ratio_ffn': ffn_ratio})

    hidden_embed = {}

    # for i, early_exit_layer in enumerate(candidate_premature_layers):
        # hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
    for i, early_exit_layer in enumerate(candidate_premature_layers):
        hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
        # hidden_embed[early_exit_layer] = [
        #     [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in hidden_states[early_exit_layer][0][pos]]
        #     for pos in range(hidden_states[early_exit_layer].shape[1])
        # ]

    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return hidden_embed, answer, activate, o_layers


# lang_set = ["english"]

def detection_all(model_name, lang, atten_ratio=0.1, ffn_ratio=0.1, test_size=-1, 
                  detection_path="./corpus_all", output_path="./output", 
                  suffix="", sample_size=1000, task="detect", args=None):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    if "gemma-3" in model_name.lower():
        # model = Gemma3ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
        model = Gemma3ForConditionalGeneration.from_pretrained(model_name, config=config, device_map="auto")
        model = model.language_model
    elif "llama" in model_name.lower():
        model = LlamaForCausalLMDetect.from_pretrained(model_name, config=config, device_map="auto")
    elif "gemma-2" in model_name.lower():
        model = Gemma2ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "qwen3" in model_name.lower():
        model = Qwen3ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "qwen" in model_name.lower():
        model = Qwen2ForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    elif "gemma" in model_name.lower():
        model = GemmaForCausalLM.from_pretrained(model_name, config=config, device_map="auto")
    else:
        raise ValueError(f"Model {model_name} not supported")
    for l in lang:
        print("Detecting neurons for", l)
        neurons = detect_key_neurons(model, tokenizer, l, atten_ratio=atten_ratio, ffn_ratio=ffn_ratio, test_size=test_size, detection_path=detection_path, output_path=output_path, suffix=suffix, model_name=model_name, sample_size=sample_size, task=task, args=args)
        print(l, "complete", len(neurons["attn_q"].keys()), len(neurons["attn_q"][0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--corpus_path", type=str, default='./dataset')
    parser.add_argument("--corpus_size", type=int, default=-1)
    parser.add_argument("--base", type=str, default="Llama-3.2-3B")
    parser.add_argument("--output_path", type=str, default="./neuron_train_data_detect/")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--atten_ratio", type=float, default=0.1)
    parser.add_argument("--ffn_ratio", type=float, default=0.1)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--task", type=str, default="mmlu")
    args = parser.parse_args()

    if ',' in args.lang:
        lang_set = args.lang.split(',')
    else:
        lang_set = [args.lang]

    detection_all(args.base, lang_set,
                  args.atten_ratio,
                  args.ffn_ratio,
                  args.corpus_size,
                  args.corpus_path,
                  args.output_path,
                  args.suffix,
                  args.sample_size,
                  args.task,
                  args)


    
    