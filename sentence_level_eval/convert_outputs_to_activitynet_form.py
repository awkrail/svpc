import json

# sentence-levelの評価をするために, convert

def convert(filename, test_recipe_id_set):
    filename, model_type = filename
    with open(filename, "r") as f:
        result = json.load(f)
    output_dict = {
            "version" : "VERSION 1.0",
            "results" : {},
            "external_data" : {
                "used" : "true",
                "details" : "global_pool layer from BN-Inception pretrained from ActivityNet \
                             and ImageNet (https://github.com/yjxiong/anet2016-cuhk)"
                }
            }

    for recipe_id, output in result["results"].items():
        timestamps = []
        sentences = []

        if recipe_id in test_recipe_id_set:
            output_dict["results"][recipe_id] = []
            for o in output:
                sentence = o["sentence"]
                timestamp = o["timestamp"]

                output_dict["results"][recipe_id].append(
                            {
                            "sentence" : sentence,
                            "timestamp" : timestamp
                            })
    
    with open("outputs/" + model_type + ".json", "w") as f:
        json.dump(output_dict, f)

if __name__ == "__main__":
    mart_ing_path = "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/outputs_2_5/mart_w_ing/mart_best_greedy_pred_val.json"
    viv_path = "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/outputs_2_5/reason_copy/reason_copy_best_greedy_pred_val.json"
    vivt_path = "/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/debbued_version/full_lambda_0.5_tau_0.5_test_greedy_pred_test.json"
    test_path = "/home/nishimura/research/recipe_generation/state_aware_VPC/proposed_recipe_generation/video_recipe_generator/captioning_model/densevid_eval/our_yc2_data/debugged_split/bosselut_split_yc2_test_anet_format.json"

    filenames = [(mart_ing_path, "mart"), (viv_path, "viv"), (vivt_path, "vivt")]
    with open(test_path, "r") as f:
        test_data = json.load(f)
    
    for filename in filenames:
        convert(filename, set(test_data.keys()))



