import json
import os
import random
import pickle
random.seed(42)

if __name__ == "__main__":
    file_root_dir = "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/outputs_2_5"
    
    with open(os.path.join(file_root_dir, "repred", "full_lambda_0.25_tau_0.5_test_greedy_pred_test.json"), "r") as f:
        repred_data = json.load(f)
    
    with open(os.path.join(file_root_dir, "mart_w_ing", "mart_best_greedy_pred_val.json"), "r") as f:
        mart_data = json.load(f)
    
    with open(os.path.join(file_root_dir, "reason_copy", "reason_copy_best_greedy_pred_val.json"), "r") as f:
        reason_data = json.load(f)
    
    with open("/home/nishimura/research/recipe_generation/graph_youcook2_generator/proposed_recipe_generation/video_recipe_generator/preprocess/split/bosselut_split_yc2_test_anet_format.json", "r") as f:
        test_annotation = json.load(f)
    
    recipe_ids = list(repred_data["results"].keys())
    ok_recipe_ids = ["hLTNXDKU_Pk",
                      "eHk6NSLGAkc",
                      "woTrhsB_bcA",
                      "We2CzpjPD3k",
                      "E9O9-6TQUw0",
                      "RY10IUcz3bk",
                      "7r6JQycloEs",
                      "cDYCtBwin5g",
                      "wlq30WwXwSM",
                      "84i8Qdnyd0k"]
    recipe_ids = list(set(recipe_ids) - set(ok_recipe_ids))
    random.shuffle(recipe_ids)
    recipe_ids = recipe_ids[:110] + ok_recipe_ids

    result_dicts = []

    for recipe_id in recipe_ids:
        mart_recipe = [x["sentence"] for x in mart_data["results"][recipe_id]]
        reason_recipe = [x["sentence"] for x in reason_data["results"][recipe_id]]
        repred_recipe = [x["sentence"] for x in repred_data["results"][recipe_id]]
        ingredients = test_annotation[recipe_id]["ingredients"]
        result_dicts.append({
            "ingredients" : ingredients,
            "recipe_id" : recipe_id,
            "recipes" : {
                "baseline" : mart_recipe,
                "half" : reason_recipe,
                "full" : repred_recipe
                }
            })
    
    with open(os.path.join(file_root_dir, "human_evaluation.pkl"), "wb") as f:
        pickle.dump(result_dicts, f)
    
