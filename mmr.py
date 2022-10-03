import collections
import torch

def doc_img_sim(x, q, ngram_embeds, image_embed):
    return torch.nn.functional.cosine_similarity(
        ngram_embeds[x].unsqueeze(0),
        image_embed.unsqueeze(0),
        dim=1
    )

def doc_doc_sim(x, y, ngram_embeds):
    return torch.nn.functional.cosine_similarity(
        ngram_embeds[x].unsqueeze(0),
        ngram_embeds[y].unsqueeze(0),
        dim=1
    )

def mmr_sorted(docs, q, lambda_, ngram_embeds, image_embed): 
    """Sort a list of docs by Maximal marginal relevance 
	 
	Performs maximal marginal relevance sorting on a set of 
	documents as described by Carbonell and Goldstein (1998) 
	in their paper "The Use of MMR, Diversity-Based Reranking 
	for Reordering Documents and Producing Summaries" 
 
    :param docs: a set of documents to be ranked  
				  by maximal marginal relevance 
    :param q: query to which the documents are results 
    :param lambda_: lambda parameter, a float between 0 and 1 
    :param similarity1: sim_1 function. takes a doc and the query 
						as an argument and computes their similarity 
    :param similarity2: sim_2 function. takes two docs as arguments 
						and computes their similarity score 
    :return: a (document, mmr score) ordered dictionary of the docs 
			given in the first argument, ordered my MMR 
    """ 
    selected = collections.OrderedDict() 
    while set(selected) != docs: 
        remaining = docs - set(selected) 
        mmr_score = lambda x: (
            lambda_*doc_img_sim(x, q, ngram_embeds, image_embed) - 
            (1-lambda_)*max([doc_doc_sim(x, y, ngram_embeds) for y in set(selected)-{x}] or [0]) 
        )
        next_selected = argmax(remaining, mmr_score) 
        selected[next_selected] = len(selected) 
    return selected 
 
def argmax(keys, f): 
    return max(keys, key=f)
