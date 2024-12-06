import t2v_metrics

#clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model
clip_flant5_score = t2v_metrics.models.vqascore_models.InstructBLIPModel(device='cpu')

### Calculate the pairwise similarity scores 
### between M images and N texts, run the following to return a M x N score tensor.
images = ["images/cat-ref.jpeg"]
texts = [
    "image of a cat"]
scores = clip_flant5_score.forward(images=images, texts=texts) # scores[i][j] is the score between image i and text j
print(scores)
