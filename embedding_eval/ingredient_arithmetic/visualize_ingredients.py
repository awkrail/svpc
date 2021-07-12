import numpy as np
import pickle

def search(query, targets):
    for target in targets:
        dist = np.sqrt(np.sum(np.square(query.numpy() - target["orig_embedding"].numpy() )))
        target["dist"] = dist
    targets = sorted(targets, key=lambda x : x["dist"])
    return targets[0]

if __name__ == "__main__":
    with open("step_emebedding_dict.pkl", "rb") as f:
        ingr_embs = pickle.load(f)
   
    with open("ingredient_embeddings_w_meta_thres05.pkl", "rb") as f:
        ingr_w_meta_dicts = pickle.load(f)
    
    for ingr_w_meta_dict in ingr_w_meta_dicts:
        recipe_id = ingr_w_meta_dict["recipe_id"]
        ingr_num, step_num = ingr_w_meta_dict["ingr_num"], ingr_w_meta_dict["step_num"]
        ingr_orig_embeddings = ingr_embs[recipe_id]
        if step_num == -1:
            ingr_w_meta_dict["orig_embedding"] = ingr_orig_embeddings["entity_vectors"][0][ingr_num]
        else:
            ingr_w_meta_dict["orig_embedding"] = ingr_orig_embeddings["entity_vectors"][1][step_num, ingr_num]

    
    add_vector = ingr_w_meta_dicts[2259]["orig_embedding"] - ingr_w_meta_dicts[2251]["orig_embedding"] # add eggs - eggs
    cut_vector = ingr_w_meta_dicts[9]["orig_embedding"] - ingr_w_meta_dicts[0]["orig_embedding"] # cut tomatoes - tomatoes
    fry_vector = ingr_w_meta_dicts[529]["orig_embedding"] - ingr_w_meta_dicts[519]["orig_embedding"] # fried onion - onion

    # multi-hop reasoning
    #added_beef = fry_vector + ingr_w_meta_dicts[163]["orig_embedding"]
    #nearest_emb = search(added_beef, ingr_w_meta_dicts)

    # cut + potatoes
    cut_potatoes = ingr_w_meta_dicts[729]["orig_embedding"] + cut_vector
    cut_nearest_emb_dict = search(cut_potatoes, ingr_w_meta_dicts)
    print("=== cut + potatoes ===")
    print(cut_nearest_emb_dict["ingredient"])
    print(cut_nearest_emb_dict["sentence"])
    print("step: ", cut_nearest_emb_dict["step_num"] + 1)
    print("recipe_id: ", cut_nearest_emb_dict["recipe_id"])
    print("======================")

    # add + tomatoes
    add_tomatoes = ingr_w_meta_dicts[0]["orig_embedding"] + add_vector
    nearest_emb_dict = search(add_tomatoes, ingr_w_meta_dicts)
    print("=== add + tomatoes ===")
    print(nearest_emb_dict["ingredient"])
    print(nearest_emb_dict["sentence"])
    print("step: ", nearest_emb_dict["step_num"] + 1)
    print("recipe_id: ", nearest_emb_dict["recipe_id"])
    print("======================")

    # add + flour
    add_flour = ingr_w_meta_dicts[308]["orig_embedding"] + add_vector
    nearest_emb_dict = search(add_flour, ingr_w_meta_dicts)
    print("=== add + flour ===")
    print(nearest_emb_dict["ingredient"])
    print(nearest_emb_dict["sentence"])
    print("step: ", nearest_emb_dict["step_num"] + 1)
    print("recipe_id: ", nearest_emb_dict["recipe_id"])
    print("======================")

    # fry + bacon
    fry_bacon = ingr_w_meta_dicts[163]["orig_embedding"] + fry_vector
    nearest_emb_dict = search(fry_bacon, ingr_w_meta_dicts)
    print("=== fry + bacon ===")
    print(nearest_emb_dict["ingredient"])
    print(nearest_emb_dict["sentence"])
    print("step: ", nearest_emb_dict["step_num"] + 1)
    print("recipe_id: ", nearest_emb_dict["recipe_id"])
    print("======================")

    # multi-hop reasoning
    # setting 1: potatoes + cut + add
    # setting 2: potatoes + cut -> retrieve + add
    nearest_emb_dict = search(cut_potatoes + add_vector, ingr_w_meta_dicts)
    print("=== cut_potatoes + add (multi-hop)/setting 1 ===")
    print(nearest_emb_dict["ingredient"])
    print(nearest_emb_dict["sentence"])
    print("step: ", nearest_emb_dict["step_num"] + 1)
    print("recipe_id: ", nearest_emb_dict["recipe_id"])
    print("======================")

    nearest_emb_dict = search(cut_nearest_emb_dict["orig_embedding"] + add_vector, ingr_w_meta_dicts)
    import ipdb; ipdb.set_trace()
    print("=== cut_potatoes + add (multi-hop)/setting 1 ===")
    print(nearest_emb_dict["ingredient"])
    print(nearest_emb_dict["sentence"])
    print("step: ", nearest_emb_dict["step_num"] + 1)
    print("recipe_id: ", nearest_emb_dict["recipe_id"])
