# %%
# Try to steer with PC1

# %%
import json
import os
import torch
import sys

sys.path.append('.')
sys.path.append('..')

from utils.steering_utils import ActivationSteering
from utils.probing_utils import *
from utils.inference_utils import *

torch.set_float32_matmul_precision('high')

# %%
layer = 22
pc_number = 6

# %%
# steering on PC1
rp_basis = torch.load('/workspace/persona-subspace/roles/pca/layer22_pos23.pt', weights_only=False)
rp_basis = rp_basis['pca'].components_[pc_number]

# %%
rp_basis = torch.from_numpy(rp_basis)
print(f"Role PC1 shape: {rp_basis.shape}")

# %%
print(rp_basis.max(), rp_basis.min())

# %%
output_dir = "/workspace/persona-subspace/roles/results/steering"

# %%
CHAT_MODEL_NAME = "google/gemma-3-12b-it"
model, tokenizer = load_model(CHAT_MODEL_NAME, device="cuda:0")

# %%
question_path = '/workspace/persona-subspace/traits/data/questions_240.jsonl'
questions = []
with open(question_path, 'r') as f:
    for line in f:
        questions.append(json.loads(line))
test_questions = questions[:1]

# %%
magnitudes = [-3000.0, 3000.0]
# magnitudes = [0]

prompts = [
    # "What are you?",
    # "How do you feel?",
    "My neighbor has bedbugs... should I be worried?"
]

test_questions = [{"question": prompt, "id": i} for i, prompt in enumerate(prompts)]

# %%
try:
    steered_results = json.load(open(f"{output_dir}/pc{pc_number}.json"))
except FileNotFoundError:
    steered_results = {}

for magnitude in magnitudes:
    print(f"\n{'='*20} Magnitude: {magnitude:+.1f} {'='*20}")
    
    try:
        with ActivationSteering(
            model=model,
            steering_vectors=rp_basis,
            coefficients=magnitude,
            layer_indices=layer,
            intervention_type="addition",
            positions="all"
        ) as steerer:
            for question in test_questions:
                prompt = question
                prompt_id = prompt['id']
                if prompt_id not in steered_results:
                    steered_results[prompt_id] = {}
                
                print(f"\nPrompt: {prompt['question']}")
                response = generate_text(model, tokenizer, prompt['question'], chat_format=True)
                print(f"Response: {response}")
                if magnitude not in steered_results[prompt_id]:
                    steered_results[prompt_id][magnitude] = []
                steered_results[prompt_id][magnitude].append(response)
    except Exception as e:
        error_msg = f"Error with magnitude {magnitude}: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise e

with open(f"{output_dir}/pc{pc_number}.json", "w") as f:
    json.dump(steered_results, f, indent=2)

# %%
model.language_model.layers
# %%
