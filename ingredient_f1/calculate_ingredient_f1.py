import os
import json
from tqdm import tqdm
"""
extract ingredients and actions from generated/ground-truth recipes
"""

def extract_ingredients(result_dict, all_ingredient_dict):
    for method, result in result_dict.items():
        for recipe_id, annotation in result.items():
            ingredient_list = annotation["ingredients"]
            sentences = annotation["sentences"]
            step_ingredient_lists = []
            for sentence in sentences:
                step_ingredient_list = []
                
                # まずは材料リストの材料を確認
                for ingredient in ingredient_list:
                    if ingredient in sentence:
                        step_ingredient_list.append(ingredient)
                
                # 次に他の材料 (test setのみ)
                for word in sentence.split(" "):
                    if word in ingredient_list:
                        continue
                    if word in all_ingredient_dict:
                        step_ingredient_list.append(word)
                

                step_ingredient_lists.append(step_ingredient_list)
            annotation["step_ingredients"] = step_ingredient_lists
    return result_dict

def calculate_ingredient_f1(result_dict, all_ingredient_dict):
    result_dict = extract_ingredients(result_dict, all_ingredient_dict)
    methods = list(result_dict.keys())[1:]
    gt_recipes = result_dict["gt"]

    for method in methods:
        
        recall_total, precision_total = 0, 0
        correct_total = 0

        for recipe_id, sentences in result_dict[method].items():
            gen_step_ingredients = sentences["step_ingredients"]
            gt_step_ingredients = gt_recipes[recipe_id]["step_ingredients"]
            for gen_ingredients, gt_ingredients in zip(gen_step_ingredients, gt_step_ingredients):
                for i, gen_ingredient in enumerate(gen_ingredients):
                    if gen_ingredient in gt_ingredients:
                        correct_total += 1
                recall_total += len(gt_ingredients)
                precision_total += len(gen_ingredients)
        
        recall = correct_total/recall_total
        precision = correct_total/precision_total

        print("------ ", method, " -----")
        print("recall: ", recall)
        print("precision: ", precision)
        print("f1: ", 2 * recall * precision / (recall + precision))
        print("-------------------------")
                    
def construct_ingredient_dict():
    root_dir = "/home/nishimura/research/recipe_generation/graph_youcook2_generator/proposed_recipe_generation/video_recipe_generator/captioning_model/densevid_eval/our_yc2_data/debugged_split"
    filenames = ["bosselut_yc2_train_anet_format.json", "bosselut_split_yc2_val_anet_format.json", "bosselut_split_yc2_test_anet_format.json"]

    all_ingredient_dict = set()

    for filename in filenames:
        with open(os.path.join(root_dir, filename), "r") as f:
            data = json.load(f)
        
        for recipe_id, annotation in data.items():
            for ingredient in annotation["ingredients"]:
                all_ingredient_dict.add(ingredient)

    return all_ingredient_dict

if __name__ == "__main__":
    # result まとめ
    result_dict = {}
    
    # gtのみ別処理
    gt_dict = {
            "gt" : "/home/nishimura/research/recipe_generation/graph_youcook2_generator/proposed_recipe_generation/video_recipe_generator/preprocess/split/bosselut_split_yc2_test_anet_format.json"
            }
    with open(gt_dict["gt"], "r") as f:
        data = json.load(f)

    
    all_ingredient_dict = construct_ingredient_dict()

    test_recipe_ids = list(data.keys())

    result_dict["gt"] = {}

    for recipe_id, annotation in data.items():
        sentences = annotation["sentences"]
        result_dict["gt"][recipe_id] = {}
        result_dict["gt"][recipe_id]["ingredients"] = annotation["ingredients"]
        result_dict["gt"][recipe_id]["sentences"] = sentences

    # その他は同じ
    filename_dict = {
            "mart" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/captioning/outputs_2_5/mart/mart_best_greedy_pred_val.json",
            "mart_w_ing" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/baselines/recurrent-transformer/features/model/w_ingredients_copy/mart_best_greedy_pred_val.json",
            #"xl" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/captioning/outputs_2_5/xl/xl_best_greedy_pred_val.json",
            "xl" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/outputs_2_5/xl/xl_tmp_greedy_pred_test.json",
            "xl_w_ing" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/baselines/recurrent-transformer/features/model/w_ingredients_copy/xl_best_greedy_pred_val.json",
            "video" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/captioning/outputs_2_5/video/video_best_greedy_pred_val.json",
            "ingr" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/captioning/outputs_2_5/ingr/ingr_best_greedy_pred_val.json",
            "copy" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/outputs_2_5/copy/copy_lambda_0.5_tau_0.5_best_greedy_pred_val.json",
            "reasoning" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/outputs_2_5/reasoning/reasoning_best_greedy_pred_val.json",
            "reason_copy" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/outputs_2_5/reason_copy/reason_copy_best_greedy_pred_val.json",
            "repred" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/debbued_version/full_lambda_0.5_tau_0.5_test_greedy_pred_test.json"
            #"repred" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/debbued_version_2/full_lambda_0.75_tau_0.5_test_greedy_pred_test.json",
            #"lambda_025" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/proposed_method/full_lambda_0.25_tau_0.5_test_greedy_pred_test.json",
            #"lambda_050" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/proposed_method/full_lambda_0.5_tau_0.5_test_greedy_pred_test.json",
            #"lambda_1" : "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/proposed_method/full_lambda_1.0_tau_0.5_test_greedy_pred_test.json"
            }
    
    for method, filename in filename_dict.items():
        result_dict[method] = {}
        with open(filename, "r") as f:
            data = json.load(f)
        for recipe_id, output in data["results"].items():
            if recipe_id in test_recipe_ids:
                result_dict[method][recipe_id] = {}
                result_dict[method][recipe_id]["ingredients"] = result_dict["gt"][recipe_id]["ingredients"]
                result_dict[method][recipe_id]["sentences"] = [o["sentence"] for o in output]

    calculate_ingredient_f1(result_dict, all_ingredient_dict)
