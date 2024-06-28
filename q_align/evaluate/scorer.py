from PIL import Image
import torch.nn as nn
import torch
import numpy as np
from typing import List
from datasets import load_dataset
from q_align.model.builder import load_pretrained_model
from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from q_align.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import requests
from io import BytesIO

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

def update_matrix(anchor_matrix, scores, indices):
    n = anchor_matrix.shape[0]
    new_row = np.zeros((1, n))
    new_col = np.zeros((n + 1, 1))
    new_row[0, indices] = scores
    new_col[indices, 0] = 1-scores  # Assuming symmetric preference for simplicity
    anchor_matrix = np.vstack([anchor_matrix, new_row])
    anchor_matrix = np.hstack([anchor_matrix, new_col])
    anchor_matrix[n, n] = 0.5
    print(anchor_matrix)
    return anchor_matrix

class Compare2Scorer(nn.Module):
    def __init__(self, pretrained="q-future/Compare2Score", device="cuda:0"):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, "mplug_owl2", device=device)
        prompt = "USER: <|image|> <|image|> Compared with the first image, what is your quality rating for second image? \nASSISTANT: The quality of the second image is"
        self.preferential_ids_ = [id_[1] for id_ in tokenizer(["inferior", "worse", "similar", "better", "superior"])["input_ids"]]
        self.anchor_images = load_dataset("VQA-CityU/Anchor_images")
        
        self.weight_tensor = np.array([0., 0.25, 0.5, 0.75, 1.], dtype=np.float16)
        self.anchor_matrix = np.array(
            [[5.0000000e-01, 2.5912809e-01, 3.3130276e-04, 1.6087297e-06, 1.1803027e-09],
             [7.4087191e-01, 5.0000000e-01, 2.4985345e-01, 9.9954158e-02, 1.8675303e-08],
             [9.9966872e-01, 7.5014657e-01, 5.0000000e-01, 4.9968880e-01, 2.4852838e-01],
             [9.9999839e-01, 9.0004587e-01, 5.0031120e-01, 5.0000000e-01, 2.5400183e-01],
             [1.0000000e+00, 1.0000000e+00, 7.5147164e-01, 7.4599814e-01, 5.0000000e-01]], 
            dtype=np.float32)
        anchor_intervals = 5#16
        num_anchor_image_per_interval = 1
        num_anchor_image = anchor_intervals * num_anchor_image_per_interval
        self.anchor_indices = np.arange(0, num_anchor_image)
        # self.preferential_ids_ = [id_[1] for id_ in tokenizer(["excellent","good","fair","poor","bad"])["input_ids"]]
        # self.weight_tensor = torch.Tensor([1,0.75,0.5,0.25,0.]).half().to(model.device)
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
    def download_image(self, url):
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert('RGB')

    def load_image(self, path):
        if path.startswith('http://') or path.startswith('https://'):
        # if "http" in path:
            return self.download_image(path)
        return Image.open(path).convert('RGB')
    
    def expand2square(self, pil_img, background_color):
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
        
    def forward(self, image_path):
        anchor_images = [item['image'] for item in self.anchor_images['train']]
        probabilities = []
        for index in self.anchor_indices:
            anchor_image = anchor_images[index]
            image = self.load_image(image_path)
            images = [anchor_image, image]
            images = [self.expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in images]
            image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(self.device)
            
            with torch.inference_mode():
                output_logits = self.model(self.input_ids, images=image_tensor)["logits"][:, -1, self.preferential_ids_]
                output_logits = output_logits.cpu().detach().numpy() / 100
                print(output_logits)
                probabilities.append(np.dot(softmax(output_logits),  self.weight_tensor))
        updated_matrix = update_matrix(self.anchor_matrix, np.squeeze(np.array(probabilities)), self.anchor_indices)
        score = optimize_score_map_pytorch_cuda(updated_matrix, seed=0, original_seed=20020, num_iterations=100)
        print(score)
        return score

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="VQA-CityU/Compare2Score_1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_path", type=str, default="/home/zhw/IQA/code/NeurIPS24/Q-Align/Compare2Score/figs/singapore_flyer.jpg")
    parser.add_argument("--aesthetic", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()


    scorer = Compare2Scorer(pretrained=args.model_path, device=args.device) 
    print(scorer(args.img_path))

