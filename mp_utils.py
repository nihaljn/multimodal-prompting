import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

import scripts.mmr as mmr


# CLIP model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
version = 'openai/clip-vit-large-patch14'
clip_model = CLIPModel.from_pretrained(version).to(device)
clip_processor = CLIPProcessor.from_pretrained(version)
clip_model.eval()

# preprocessed prompt data
print('Loading ngram prompt data...', end=' ', flush=True)
with open('data/ngram_prompts.txt', 'r') as f:
    all_prompts = [l.strip() for l in f.readlines()]
ngram_prompt_embeds = torch.from_numpy(
    np.load('data/clip_text_embed.npy')
).to(device)
print('Done.')


def add_mp_arguments(parser):
    parser.add_argument(
        "--prompt-images",
        type=str,
        help="comma separated paths to images that are used in the prompt",
        default=''
    )
    parser.add_argument(
        "--image-weights",
        type=str,
        help="comma separated weights of images that are used in the prompt. \
              the weights should ideally be in [1, 10]; 4 is the default value",
        default=''
    )


def preprocess_prompt(opt):
    '''Break opt.prompt into list of strings
    so that opt.prompt_images can be added between them.
    Fuction returns a list of interleaved string and image embeds'''

    parts = re.split('\[|\]', opt.prompt)
    parts = list(filter(None, parts))

    ret_parts = []
    image_count = 0

    for part in parts:
        if part.startswith('img'):
            if len(opt.prompt_images) == image_count:
                raise ValueError('# of prompt images incompatible with prompt')
            image_path = opt.prompt_images[image_count]
            image = Image.open(image_path)
            image_embeds = clip_model.get_image_features(
                **clip_processor(images=[image], return_tensors='pt').to(device)
            )[0]
            ret_parts.append(image_embeds)
            image_count += 1
        else:
            ret_parts.append(part)

    assert image_count == len(opt.prompt_images), (
        '# of prompt images incompatible with prompt'
    )

    return ret_parts


def combine_str_prompts(strs):
    prompts = list(map(lambda x: x.strip(), strs))
    prompt = ', '.join(prompts)
    return prompt


def concatenated_conditioning(prompt, model, batch_size, weights):
    '''form the conditioning by concatenating strings (even for images)'''
    combined_prompt = ''
    img_cnt = 0
    for j, prompt_part in enumerate(prompt):
        if isinstance(prompt_part, str):
            combined_prompt += ' ' + prompt_part
        else:
            sims = torch.nn.functional.cosine_similarity(
                ngram_prompt_embeds, prompt_part.unsqueeze(0), dim=1
            )
            cand_indices = torch.sort(sims, descending=True)[1][:100].cpu()
            sorted_cand_indices = mmr.mmr_sorted(
                set(cand_indices), None, 0.7, ngram_prompt_embeds, prompt_part
            )
            selected_prompts = [all_prompts[i] for i in sorted_cand_indices]
            # combined_prompt += ' ' + combine_str_prompts(selected_prompts[:5])
            combined_prompt += ' '.join(selected_prompts[:weights[img_cnt]])
            img_cnt += 1
    prompts = [combined_prompt] * batch_size
    print(combined_prompt)
    c = model.get_learned_conditioning(prompts)
    return c


def pooled_conditioning(prompt, model, batch_size, max_length=77):
    '''form the conditioning by pooling the various texts for each image.
    Note: not tested thoroughly.'''
    part_reps = []
    for prompt_part in prompt:
        if isinstance(prompt_part, str):
            clip_inps_str = [prompt_part]
        else:
            sims = torch.nn.functional.cosine_similarity(
                ngram_prompt_embeds, prompt_part.unsqueeze(0), dim=1
            )
            cand_indices = torch.sort(sims, descending=True)[1][:100].cpu()
            sorted_cand_indices = mmr.mmr_sorted(
                set(cand_indices), None, 0.7, ngram_prompt_embeds, prompt_part
            )
            selected_prompts = [all_prompts[i] for i in sorted_cand_indices][:5]
            clip_inps_str = [' '.join(selected_prompts)]
        clip_inps = clip_processor(
            text=clip_inps_str, return_tensors='pt', padding='max_length',
            max_length=max_length, truncation=True
        ).to(device)
        cur_reps = model.cond_stage_model.transformer(**clip_inps).last_hidden_state
        # attn = clip_inps.attention_mask
        # cur_reps = ((cur_reps * attn.unsqueeze(-1)).sum(1) / attn.sum(-1).unsqueeze(-1))
        part_reps.append(cur_reps)
    weights = torch.tensor([100, 200, 200]).view(-1, 1, 1).to(device)
    c = torch.cat(part_reps, dim=0) * weights
    c = c.sum(dim=0, keepdims=True) / weights.sum()
    c = c.repeat(batch_size, 1, 1)
    return c


def get_conditional_embedding(prompt, model, batch_size, mode, weights):
    if mode == 'concat':
        return concatenated_conditioning(prompt, model, batch_size, weights)
    else:
        return pooled_conditioning(prompt, model, batch_size)

'''

def process(strs):
    prompts = list(map(lambda x: x.strip(), strs))
    prompt = ', '.join(prompts)
    return prompt
# print(prompts)
# prompts = ['a garden with flowers',
# 'accurate krita digital masterpiece',
# 'commissioned street scene',
# 'cyberpunk Tokyo market stall',
# 'artstation matte painting']
prefix = ['A garden full of colorful flowers']
prefix = process(prefix)
suffix = ['neon digital art trending ']
suffix = process(suffix)
# prompt = f'{prefix} IN THE STYLE OF {suffix}'
prompt = f'{prefix} with lights from {suffix}'
print(prompt)
prompts = [prompt] * 3

# weights = torch.tensor([1000, 20, 50, 150, 60]).to(device)



# c = c * weights.view(-1, 1, 1)
# c = c.sum(dim=0, keepdim=True).repeat(3, 1, 1) / weights.sum()
# exit(0)

# version = 'openai/clip-vit-large-patch14'
# transformer = CLIPModel.from_pretrained(version)
# processor = CLIPProcessor.from_pretrained(version)

# inputs = processor(
#     text=prompts, return_tensors="pt", padding=True
# )

# outputs = transformer.get_text_features(**inputs)
# outputs = outputs.unsqueeze(1).repeat(1, 77, 1)
# c = outputs.to(device)

'''