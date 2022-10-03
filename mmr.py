import torch

def mmr_sorted(docs, lambd, ngram_embeds, image_embed): 
    """Sort a list of docs by Maximal marginal relevance (MMR)
	Performs maximal marginal relevance sorting on a set of 
	documents as described by Carbonell and Goldstein (1998) 
	in their paper "The Use of MMR, Diversity-Based Reranking 
	for Reordering Documents and Producing Summaries".
    Args
        ngram_embeds: torch.Tensor of size (num_docs, embed_dim)
        image_embed: torch.Tensor of size (embed_dim)
        lambd: float weight given to relevance relative (on scale of 1.) to diversity
    Returns
        selected: docs resorted by MMR score""" 
    
    selected = []

    ngram_embeds = ngram_embeds[list(docs)]
    dtoi = {d: i for i, d in enumerate(list(docs))}
    doc_img_sims = torch.nn.functional.cosine_similarity(
        ngram_embeds,
        image_embed.unsqueeze(0),
        dim=1
    )
    doc_doc_sims = torch.nn.functional.cosine_similarity(
        ngram_embeds.unsqueeze(0),
        ngram_embeds.unsqueeze(1), 
        dim=-1
    )

    while len(selected) != len(docs): 
        remaining = docs - set(selected)
        mmr_score = lambda x: (
            lambd*doc_img_sims[dtoi[x]] - 
            (1-lambd)*max([doc_doc_sims[dtoi[x]][dtoi[y]] for y in set(selected)-{x}] or [0]) 
        )
        next_selected = argmax(remaining, mmr_score) 
        selected.append(next_selected)

    return selected
 

def argmax(keys, f): 
    return max(keys, key=f)


'''
References
----------
https://qr.ae/pvaEse
'''