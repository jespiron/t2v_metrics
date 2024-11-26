import t2v_metrics
import numpy as np
from typing import List

clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # the recommended VQA scoring model
clip_score = t2v_metrics.CLIPScore(model='openai:ViT-L-14-336')   # classic CLIP scoring

### Calculate similarity score for a single (image, text) pair
###     "Does this image show {text}? Yes/No"
### Returns: a singleton score tensor.
###     Higher score indicates Yes, lower score indicates No.
def calculate_similarity_pair(image_path: str, text_prompt: str):
    score = clip_flant5_score(images=[image_path], texts=[text_prompt])
    return score

### Calculates the pairwise similarity scores between M images and N texts
### Returns: a M x N score tensor, scores
###             scores[i][j] is the score between image i and text j
def calculate_similarity_pairs(image_paths: List[str], text_prompts: List[str]):
    scores = clip_flant5_score(images=image_paths, texts=text_prompts)
    return scores

###
### To summarize our task,
###     We have a batch of N images that represent text prompt A.
###     We edit these N images to represent text prompt B.
###
### With our task in mind, the metrics I think are worth measuring are:
###     (1) To what extent are these N edited images similar to each other?
###     (2) To what extent do these N images represent text prompt B?
###     (3) To what extent do these N images remain faithful to the original images?
###
### For (1), I have `calculate_intrasimilarity`,
### For (2), I have `calculate_representation_accuracy`
### For (3), I have a `calculate_faithfulness`
###
def calculate_intrasimilarity(image_paths: List[str], text_prompt,
                                pairwise_similarity_scores=None):
    if pairwise_similarity_scores is None:
        pairwise_similarity_scores = calculate_similarity_pairs(image_paths, [text_prompt])
    intrasimilarity_metrics = {
        'geometric_mean': np.geometric_mean(pairwise_similarity_scores),
        'variance': np.var(pairwise_similarity_scores),
        'median': np.median(pairwise_similarity_scores)
    }
    return intrasimilarity_metrics

def calculate_representation_accuracy(image_paths: List[str], text_prompt,
                                        pairwise_similarity_scores=None):
    if pairwise_similarity_scores is None:
        pairwise_similarity_scores = calculate_similarity_pairs(image_paths, [text_prompt])
    representation_metrics = {
        'mean_similarity': np.mean(pairwise_similarity_scores),
        'min_similarity': np.min(pairwise_similarity_scores),
        'distribution_coverage': {
            'median': np.median(pairwise_similarity_scores),
            'percentile_low': np.percentile(pairwise_similarity_scores, 10),
            'percentile_high': np.percentile(pairwise_similarity_scores, 90)
        }
    }
    return representation_metrics

def calculate_faithfulness(original_images, edited_images):
    return clip_score(original_images, edited_images)

def run_benchmarks(original_images, edited_images, edited_prompt):
    pairwise_similarity_scores = calculate_similarity_pairs(edited_images, [edited_prompt])

    # Run intrasimilarity benchmarks
    intrasimilarity_metrics = calculate_intrasimilarity(edited_images, edited_prompt, pairwise_similarity_scores)
    print(f"Intrasimilarity between edited images: {intrasimilarity_metrics}")

    # Run representation accuracy benchmarks
    representation_metrics = calculate_representation_accuracy(edited_images, edited_prompt, pairwise_similarity_scores)
    print(f"Representation accuracy of edits: {representation_metrics}")

    # Run faithfulness benchmarks
    faithfulness_metrics = calculate_faithfulness(edited_images, original_images)
    print(f"Faithfulness to original images: {faithfulness_metrics}")

### Demo of the benchmarks above
original_images = ["images/0.png", "images/1.png"]
edited_images = original_images
texts = ["someone talks on the phone angrily while another person sits happily",
        "someone talks on the phone happily while another person sits angrily"]
run_benchmarks(original_images, edited_images, texts[1])