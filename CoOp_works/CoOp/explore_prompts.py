import os
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip



def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


parser = argparse.ArgumentParser()
# examples like: /content/Capstone-ood/CoOp_works/CoOp/output/caltech101/CoOp/rn50_8shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200
parser.add_argument("mpath", type=str, help="the saved model to parse")
args = parser.parse_args()

mpath = args.mpath
assert os.path.exists(mpath)
print("Parsing model at ", mpath)

prompt_learner = torch.load(mpath, map_location="cpu")["state_dict"]
ctx = prompt_learner["ctx"]
ctx = ctx.float()
print(ctx)
print(f"Size of context: {ctx.shape}")

directory_path = 'explores/'
if not os.path.exists(directory_path):
    # If it doesn't exist, create the directory
    os.makedirs(directory_path)


print()
print("Computing pairwise cos similarities... ")
# Compute pairwise cosine similarity between all prompts
similarity_matrix = torch.zeros((ctx.shape[0], ctx.shape[0]))
for i in range(ctx.shape[0]):
    for j in range(i, ctx.shape[0]):  # Include diagonal and upper triangle
        similarity = cosine_similarity(ctx[i], ctx[j], dim=0)
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # Symmetric matrix

# Convert to NumPy for visualization
similarity_matrix_np = similarity_matrix.numpy()

# Plot the heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix_np, annot=True, cmap="coolwarm", cbar=True, 
            xticklabels=range(1, ctx.shape[0]+1), yticklabels=range(1, ctx.shape[0]+1))
plt.title("Cosine Similarity Heatmap Between Prompts")
plt.xlabel("Prompt Index")
plt.ylabel("Prompt Index")
file_path = os.path.join(directory_path, 'cosine_similarity_heatmap.png')
print(f"Saving the plot to '{file_path}'.")
plt.savefig(file_path)


print()
print("Computing PCA... ")
# Convert the tensor to a NumPy array and apply PCA
ctx_np = ctx.numpy()
pca = PCA(n_components=2)
ctx_pca = pca.fit_transform(ctx_np)

# Plot the 2D representation
plt.figure(figsize=(10, 8))
plt.scatter(ctx_pca[:, 0], ctx_pca[:, 1])
plt.title("PCA of ctx Prompts")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
file_path = os.path.join(directory_path, 'PCA.png')
print(f"Saving the plot to '{file_path}'.")
plt.savefig(file_path)

print()
print("Investigating Semantic Meaning, i.e. examine which tokens or phrases each prompt is most similar to in embedding space...")
print(f"Return the top-3 matched words")
tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight
print(f"Size of token embedding: {token_embedding.shape}")


if ctx.dim() == 2:
    # Generic context
    distance = torch.cdist(ctx, token_embedding)
    print(f"Size of distance matrix: {distance.shape}")
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :3]

    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        print(f"{m+1}: {words} {dist}")

elif ctx.dim() == 3:
    # Class-specific context
    raise NotImplementedError
