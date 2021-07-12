import os
import json

if __name__ == "__main__":
    result_root_dir = "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/outputs_2_5"
    method_path_dict = {
            "lambda_025" : os.path.join(result_root_dir, "repred/full_lambda_0.25_tau_0.5_test_greedy_pred_test.json"),
            "lambda_05" : os.path.join(result_root_dir, "repred/full_lambda_0.5_tau_0.5_test_greedy_pred_test.json"),
            "lambda_075" : os.path.join(result_root_dir, "repred/full_lambda_0.75_tau_0.5_test_greedy_pred_test.json"),
            "lambda_10" : os.path.join(result_root_dir, "repred/full_lambda_1.0_tau_0.5_test_greedy_pred_test.json"),
            "ground_truth" : os.path.join(result_root_dir, "ground_truth/bosselut_yc2_val_anet_format.json")
            }

    with open("/home/nishimura/research/recipe_generation/graph_youcook2_generator/proposed_recipe_generation/video_recipe_generator/preprocess/split/yc2_split_test_anet_format_para.json", "r") as f:
        test_para_data = json.load(f)
    test_recipe_ids = list(test_para_data.keys())
    
    # for task
    with open("/mnt/LSTA5/data/common/recipe/youcook2/annotations/youcookii_annotations_trainval.json", "r") as f:
        orig_annotation_data = json.load(f)

    for method, filename in method_path_dict.items():
        with open(filename, "r") as f:
            data = json.load(f)
        outputs = [["end", "start", "task", "text", "video_id"]]
        if method != "ground_truth":
            for test_recipe_id in test_recipe_ids:
                for annotation in data["results"][test_recipe_id]:
                    sentence, timestamp = annotation["sentence"], annotation["timestamp"]
                    task = orig_annotation_data["database"][test_recipe_id]["recipe_type"]
                    outputs.append([timestamp[1], timestamp[0], task, sentence, test_recipe_id])
        else:
            for test_recipe_id in test_recipe_ids:
                sentences = data[test_recipe_id]["sentences"]
                timestamps = data[test_recipe_id]["timestamps"]
                task = orig_annotation_data["database"][test_recipe_id]["recipe_type"]
                for sentence, timestamp in zip(sentences, timestamps):
                    outputs.append([timestamp[1], timestamp[0], task, sentence, test_recipe_id])

        with open(os.path.join(result_root_dir,"repred", method + "_generated_test_recipe.csv"), "w") as f:
            for output in outputs:
                line = ",".join([str(x) for x in output]) + "\n"
                f.write(line)
