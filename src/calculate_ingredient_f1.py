import os
import json
import argparse
from tqdm import tqdm

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
    root_dir = "densevid_eval/yc2_data"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="model_name for ingredient prediction evaluation (e.g., vivt, viv)")
    parser.add_argument("--caption_path", type=str, required=True, help="caption path")
    args = parser.parse_args()

    result_dict = {}
    gt_dict = { "gt" : "densevid_eval/yc2_data/bosselut_split_yc2_test_anet_format.json" }
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
    filename_dict = { args.model_name : args.caption_path }
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
