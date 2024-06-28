import argparse
import torch
import numpy as np
from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from q_align.conversation import conv_templates, SeparatorStyle
from q_align.model.builder import load_pretrained_model
from q_align.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import json
from tqdm import tqdm
from collections import defaultdict
import os
from q_align.evaluate.correlation import cal_plcc_srcc_rmse
import itertools
import random
from datasets import load_dataset
import time
# random.seed(19950801)  # Fixed seed for reproducibility
# Create a random number generator
rng = np.random.default_rng()

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)





def norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


def optimize_score_map_pytorch_cuda(c, seed=0, original_seed=20020, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    c = torch.tensor(c, dtype=torch.float32, device=device, requires_grad=False)
    initial_scores = torch.rand(c.shape[0], device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([initial_scores], lr=0.1)

    for _ in range(num_iterations):
        optimizer.zero_grad()
        sum_log_diff = torch.sum(c * torch.log(torch.maximum(torch.sigmoid(initial_scores[:, None] - initial_scores), torch.tensor(1e-6, device=device))))
        sum_squares = torch.sum(initial_scores ** 2) / 2

        loss = -(sum_log_diff - sum_squares)
        loss.backward()
        optimizer.step()
    
    optimized_scores = initial_scores.detach().cpu().numpy()
    min_score, max_score = np.min(optimized_scores), np.max(optimized_scores)
    
    # Scale scores to 0-100
    scaled_scores = 100 * (optimized_scores - min_score) / (max_score - min_score)
    
    # Reset the seed
    np.random.seed(original_seed)
    return scaled_scores[-1]

def softmax(logits):
    # exp_logits = np.exp(logits - np.max(logits))
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return probs
    # return exp_logits / exp_logits.sum()

def update_matrices(preference_matrix, scores, indices):
    n = preference_matrix.shape[0]
    new_row = np.zeros((1, n))
    new_col = np.zeros((n + 1, 1))
    new_row[0, indices] = scores
    new_col[indices, 0] = 1-scores  # Assuming symmetric preference for simplicity
    preference_matrix = np.vstack([preference_matrix, new_row])
    preference_matrix = np.hstack([preference_matrix, new_col])
    preference_matrix[n, n] = 0.5
    return preference_matrix

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
                        

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    #KonIQ-10k
    anchor_dataset = load_dataset("VQA-CityU/Anchor_images")
    anchor_matrix = np.array(
            [[5.0000000e-01, 2.5912809e-01, 3.3130276e-04, 1.6087297e-06, 1.1803027e-09],
             [7.4087191e-01, 5.0000000e-01, 2.4985345e-01, 9.9954158e-02, 1.8675303e-08],
             [9.9966872e-01, 7.5014657e-01, 5.0000000e-01, 4.9968880e-01, 2.4852838e-01],
             [9.9999839e-01, 9.0004587e-01, 5.0031120e-01, 5.0000000e-01, 2.5400183e-01],
             [1.0000000e+00, 1.0000000e+00, 7.5147164e-01, 7.4599814e-01, 5.0000000e-01]], 
            dtype=np.float32)
    anchor_intervals = 5#16
    num_anchor_image_per_interval = 1
    num_anchor_image = anchor_intervals * num_anchor_image_per_interval
    anchor_indices = np.arange(0, num_anchor_image)
    image_path = "/home/zhw/IQA/code/NeurIPS24/Q-Align/playground/data/"

    json_prefix =f"playground/data/test_jsons/1/"
    jsons = [
        json_prefix + "live_test.json",
        json_prefix + "clive_test.json",
        json_prefix + "csiq_test.json",
        json_prefix + "koniq10k_test.json",
        json_prefix + "bid_test.json",
        json_prefix + "kadid10k_test.json",
        json_prefix + "spaq_testing_set.json",
        json_prefix + "tid_testing_set.json",
        json_prefix + "agi.json",
    ]
    


    os.makedirs(f"results/{args.model_path}/", exist_ok=True)


    conv_mode = "mplug_owl2"

    inp = "<|image|> <|image|> Compared with the first image, what is your quality rating for second image?"

    conv = conv_templates[conv_mode].copy()
    # inp =  inp + "\n" + DEFAULT_IMAGE_TOKEN
    # inp = DEFAULT_IMAGE_TOKEN + inp 
    conv.append_message(conv.roles[0], inp)
    image = None
        
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + "The quality of the second image is "#

    toks = ["inferior", "worse", "similar", "better", "superior"]
    print(toks)
    ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
    print(ids_)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    
    for json_ in jsons:
        with open(json_) as f:
            iqadata = json.load(f)  

            image_tensors = []
            batch_data = []
            gt_scores = []  
            pre_soft_score = []
            for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):
                try:
                    filename = llddata["image"]
                except:
                    filename = llddata["img_path"]
                gt_score = llddata["gt_score"]
                llddata["logits"] = defaultdict(float)

                probabilities = []
                
                anchor_images = [item['image'] for item in anchor_dataset['train']]
                for index in anchor_indices:
                    anchor_image = anchor_images[index]
                    # ref_image = ref_data[index]["image"]
                    dst_image_path = os.path.join(image_path, filename)
                    images = [anchor_image, Image.open(os.path.join(dst_image_path)).convert('RGB')]
                    # image = [Image.open(os.path.join(im)).convert('RGB') for im in image_paths]
                    images = [expand2square(img, tuple(int(x*255) for x in image_processor.image_mean)) for img in images]
                    image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(args.device)

                    image_tensors.append(image_tensor)
                    batch_data.append(llddata)
                
                    with torch.inference_mode():
                        output_logits = model(input_ids, images=image_tensor)["logits"][:,-1]
 

                    for j, xllddata in enumerate(batch_data):
                        for tok, id_ in zip(toks, ids_):
                            xllddata["logits"][tok] += output_logits[j,id_].item()
                        # print(llddata)
                        json_ = json_.replace("combined/", "combined-")
                        with open(f"results/{args.model_path}/{json_.split('/')[-1]}", "a") as wf:
                            json.dump(xllddata, wf)
                        comparison = xllddata
                        t = 100
                        logits = np.array([comparison["logits"]["inferior"]/t, comparison["logits"]["worse"]/t, comparison["logits"]["similar"]/t, comparison["logits"]["better"]/t, comparison["logits"]["superior"]/t])
                        probability = softmax(logits)
                        preference = np.inner(probability, np.array([0,0.25,0.5,0.75,1.]))
                        probabilities.append(preference)

                    image_tensors = []
                    batch_data = []


                updated_matrix = update_matrices(anchor_matrix, np.array(probabilities), anchor_indices)
                # print("Preference matrix construction complete.")
                pred_score = optimize_score_map_pytorch_cuda(updated_matrix, seed=0, original_seed=20020, num_iterations=50)

                print("soft_map_result_score 100 is: ", pred_score)
                
                pre_soft_score.append(pred_score)
        
                gt_scores.append(gt_score) 
            with open(f"results/{args.model_path}/"+args.result_name, 'a') as f:
                cc, srcc, rmse = cal_plcc_srcc_rmse((np.array(gt_scores)).astype(np.float64), (np.array(pre_soft_score)).astype(np.float64))
                print("The PLCC: {}, SRCC: {} of using the soft preference matrix of the {} on {}".format(cc, srcc, args.model_path, json_.split("/")[-1]), file=f)

               
               




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='VQA-CityU/Compare2Score_1')
    parser.add_argument("--result-name", type=str, default="result.txt")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)