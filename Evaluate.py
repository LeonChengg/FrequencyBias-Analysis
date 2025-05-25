import json
import os
import argparse
import numpy as np
import random
import torch
from peft import PeftModel, LoraConfig, get_peft_model
import transformers
import textwrap
from typing import Union
from termcolor import colored
from transformers import LlamaTokenizer, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, GenerationConfig, pipeline, set_seed
from transformers.generation.utils import GreedySearchDecoderOnlyOutput

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPLATE = "Question: If [P], then [H]. Is that true or false? (A) True; (B) False;\n"

deepseek_instruct= "Direcly give me the choice. "


FEW_SHOTS = [ "Question: If Google bought Youtube, then Google owns Youtube. Is that true or false? (A) True; (B) False;\nAnswer: (A) True.\nYes, it is true. Google bought Youtube entails Google owns Youtube.",
              "Question: If Google owns Youtube, then Google bought Youtube. Is that true or false? (A) True; (B) False;\nAnswer: (B) False.\nNo, it is false. Google owns Youtube does not entail Google bought Youtube.",
              "Question: If John went to the mall, then John drove to the mall. Is that true or false? (A) True; (B) False;\nAnswer: (B) False.\nNo, it is false. John went to the mall does not entail John drove to the mall.",
              "Question: If John drove to the mall, then John went to the mall. Is that true or false? (A) True; (B) False;\nAnswer: (A) True.\nYes, it is true. John drove to the mall entails John went to the mall.",
            ]



fewshots_text = "\n".join(FEW_SHOTS) + "\n"
fewshots_hypothesis = "\n".join(FEW_SHOTS_HYPOTHESIS_tri) + "\n"


#option_indices = {'A': 319, 'B': 350, 'C': 315, 'a': 263, 'b': 289, 'c': 274, 'Entailment': 4284, 'entailment': 875,
#                  'Neutral': 2448, 'neutral': 21104, 'true': 1565, 'True': 5852, 'unknown': 9815, 'Unknown': 853, 'false': 2089, 'False': 7700, "(A":9999999, "(B":9999999}  # for multi-token words like ``entailment'', we take only the first token for scores.

option_indices = {'A': 32, 'B': 33, 'C': 34, 'a': 64, 'b': 65, 'c': 66, 'Entailment': [2300, 607, 479], 'entailment': [306, 607, 479], 'Neutral': 88007, 'neutral': 60668, 'true': 1904, 'True': 2575, 'unknown': 16476, 'Unknown': 14109, 'false': 3934, 'False': 4139, "(A":4444, "(B":5462, "(C":3100, "(a":2948, "(b":1921, "(c":1361}  # llama

#option_indices = {'A': 1098, 'B': 1133, 'C': 1102, 'a': 1032, 'b': 1055, 'c': 1045, 'Entailment': [7430, 1382, 1234], 'entailment': [1704, 1382, 1234], 'Neutral': [3915, 1097, 2418], 'neutral': 14982, 'true': 1900, 'True': 6878, 'unknown': 9806, 'Unknown': [1703, 5485], 'false': 2109, 'False': 9018, "(A":[1093, 29509], "(B":[1093, 29528], "(C":[1093, 29511], "(a":[1093, 29476], "(b":[1093, 29494], "(c":[1093, 29485]}  # Mistral


def read_data(path):
    with open(path, "r") as f:
        js = json.loads(f.read())
    data = []
    for i in js:
        hypo, prem = i["hypo"], i["prem"] 
        if args.run_data == "ori":
            #input = fewshots_text + deepseek_instruct + TEMPLATE.replace("[P]", prem[:-1]).replace("[H]", hypo[:-1])
            input = fewshots_text + TEMPLATE.replace("[P]", prem[:-1]).replace("[H]", hypo[:-1]) +"\nAnswer: "
        elif args.run_data == "contra":
            neg_hypo, neg_prem  = i["neg_hypo"], i["neg_prem"]
            input = fewshots_text + TEMPLATE.replace("[P]", neg_hypo[:-1]).replace("[H]", neg_prem[:-1])
        data.append(input)
    labels = [i["label"] for i in js]
    return data, labels

def get_scores(curr_inputs, tokenizer, curr_outputs):
    def judgement(net_scores, net_charlist, tokenizer):
        curr_pred = None
        curr_scr = None
        output_irregular_flag = False
        output_tokens = tokenizer.decode(net_charlist).lower()
        #for i in range(Length-1, 0, -1):
        for i in range(min(len(net_scores),15)):
            print(i)
            if net_scores[i] < 0:
                print(f"Warning: net score is negative: {net_scores[i]}", file=sys.stderr)
                net_scr = 0
            else:
                net_scr = net_scores[i]
            sigmoid_scr = net_scr / (1 + net_scr)
            print(tokenizer.decode(net_charlist[i]))
            #if net_charlist[i] in [option_indices['A'], option_indices['(a'], option_indices['(A'], option_indices['True'], option_indices['true']]:
            if net_charlist[i] in [option_indices['(a'], option_indices['(A'], option_indices['True'], option_indices['true']]:
                print(colored(11, "red"))
                curr_pred = 'A'
                curr_scr = 0.5 + 0.5 * sigmoid_scr
                break
            #elif net_charlist[i] in [option_indices['B'], option_indices['(b'], option_indices['(B'], option_indices['False'], option_indices['false']]:
            elif net_charlist[i] in [option_indices['(b'], option_indices['(B'], option_indices['Unknown'], option_indices['Unknown']]: ## for hypo-only
                print(colored(22,"red"))
                curr_pred = 'B'
                curr_scr = 0.5 - 0.5 * sigmoid_scr
                break
            #elif net_charlist[i] in [option_indices['C'], option_indices['(c'], option_indices['Unknown'], option_indices['unknown']]:
            elif net_charlist[i] in [option_indices['(c'], option_indices['(C'], option_indices['False'], option_indices['false']]:  ## for hypo-only
                print(colored(33,"red"))
                curr_pred = 'C'
                curr_scr = 0.5 - 0.5 * sigmoid_scr
                break
            else:
                pass

        if curr_pred is not None or curr_scr is not None:
            assert curr_pred is not None and curr_scr is not None
        else:
            if net_scores[0] < 0:
                print(f"Warning: net score is negative: {net_scores[0]}", file=sys.stderr)
                net_scr = 0
            else:
                net_scr = net_scores[0]
            sigmoid_scr = net_scr / (1 + net_scr)
            
            if (("true" in output_tokens.lower()) and ("false" not in output_tokens.lower())) or (("(a)" in output_tokens.lower()) and ("(b)" not in output_tokens.lower())):
                print(3)
                curr_pred = 'A'
                curr_scr = 0.5 + 0.5 * sigmoid_scr
            elif (("false" in output_tokens.lower()) and ("true" not in output_tokens.lower())) or (("(b)" in output_tokens.lower()) and ("(a)" not in output_tokens.lower())):
                print(4)
                curr_pred = 'B'
                curr_scr = 0.5 - 0.5 * sigmoid_scr
            elif output_tokens.startswith('true') and len(net_scores) > 0:
                curr_pred = 'A'
                curr_scr = 0.5 + 0.5 * sigmoid_scr
            elif output_tokens.startswith('false') and len(net_scores) > 0:
                curr_pred = 'B'
                curr_scr = 0.5 - 0.5 * sigmoid_scr
            else:
                print(f"Irregular output: {output_tokens}; {net_charlist}")
                output_irregular_flag = True
                curr_pred = 'B'
                curr_scr = 0.0
            
            
        if not -0.00001 <= curr_scr <= 1.00001:
            print(f"Error!!!!!!!!!!!!!!!!!!! CURR_SCR OUT OF RANGE: {curr_scr}", file=sys.stderr)

        return curr_pred, curr_scr, output_tokens, output_irregular_flag
    
    total_outlists = curr_outputs.sequences.tolist()
    net_scores = curr_outputs.scores  # seq_len * (batch_size, vocab_size)
    
    for inbatch_eidx in range(len(total_outlists)):
        this_net_outlist = []
        this_outlist = total_outlists[inbatch_eidx] # the sentence id
        for i in range(len(this_outlist)):
            ##if i < len(curr_inputs['input_ids'][inbatch_eidx]):
            if i < len(curr_inputs[inbatch_eidx]):
                assert this_outlist[i] == curr_inputs[inbatch_eidx][i].item()
            else:
                this_net_outlist.append(this_outlist[i])
        assert len(this_net_outlist) <= len(net_scores), f"{len(this_net_outlist)} vs {len(net_scores)}"
        this_net_scores = [net_scores[i][inbatch_eidx][this_net_outlist[i]].item() for i in range(len(this_net_outlist))]
        
        ## just check the first 12 or last 12 tokens
        
        Length = len(this_net_scores)
        print(Length)
        
        curr_pred, curr_scr, curr_outtokens, output_irregular_flag = judgement(this_net_scores, this_net_outlist, tokenizer)
    return curr_pred, curr_scr, curr_outtokens, output_irregular_flag

     
def run(args):
    model_id = args.model_path
    peft_model_id = args.tuned_model

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.use_tuned_model:
        model.load_adapter(peft_model_id)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    data, labels = read_data(args.data_path)
    output = []
    for i in range(len(data)):
        input_text = data[i]
        label = labels[i]
        messages = [{"role": "user", "content": input_text}]
        print(colored(input_text, "green"))
        print(colored(label, "green"))
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=48,
            eos_token_id=terminators,
            do_sample=False,
            temperature=None,
            top_p=None,
            ##**tokenized_text,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        curr_pred, curr_scr, curr_outtokens, output_irregular_flag = get_scores(input_ids, tokenizer, outputs)
        curr_outtokens = curr_outtokens.strip()
        print(colored(curr_outtokens, "blue"))
        result = {"curr_pred":curr_pred,"curr_scr":curr_scr,"curr_outtokens":curr_outtokens, "output_irregular_flag":output_irregular_flag}
        print(result)
        print("\n\n")
        
        output.append({"input_text":input_text, "label":label, "pred":result, "model":model_id.split("/")[-2], "use_tuned_model":args.use_tuned_model})
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)
        
                                   
    
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="~data/levyholt/levy_holt/lvholt_test_llama2.json")
    parser.add_argument('--model_path', type=str, default="~models/Meta-Llama-3-70B-Instruct/")
    parser.add_argument('--tuned_model', type=str, default="~llama-recipes/tuned_model_ori/llama-3-70B-EG_weed-balanced/")
    parser.add_argument('--output', type=str, default="results/")
    parser.add_argument('--run_data', type=str, choices=['ori', 'contra'])
    parser.add_argument('--use_tuned_model', action="store_true")
    parser.add_argument('--fewshot', action="store_true")
    parser.add_argument('--add_contra_prompt', action="store_true")
    
    args = parser.parse_args()
    print(args)
    
    run(args)
