import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(matrix, name):
    plt.clf()
    matrix = matrix.cpu().numpy()
    sns.heatmap(matrix, annot=True, fmt="1.2f", cmap='Reds', linewidths=.5, linecolor='black', vmax=1., vmin=0.)
    plt.savefig(name)

if __name__ == "__main__":
    with open("/mnt/LSTA5/data/nishimura/graph_youcook2_generator/baselines/recurrent-transformer/features/model/wo_ingredients/output.pkl", "rb") as f:
        data = pickle.load(f)

    attention_dicts = data["attention_dicts"]
    batch_raw_infos = data["batch_raw_infos"]

    num_arrays = []

    for batch_idx, batch_raw_info in enumerate(batch_raw_infos):
        for jdx in range(len(batch_raw_info)):
            raw_info = batch_raw_info[jdx]
            if 4 <= len(raw_info["gt_sentence"]) <= 6:
                num_arrays.append((batch_idx, jdx))
    
    for num_array in num_arrays:
        idx, jdx = num_array
        i2s_prob = attention_dicts[idx]["ingr2step"][jdx]
        s2s_prob = attention_dicts[idx]["step2step"][jdx]

        i2s_prob = i2s_prob.mean(dim=0)
        s2s_prob = s2s_prob.mean(dim=0)
        s2s_prob = s2s_prob.triu()[:-1]
        raw_info = batch_raw_infos[idx][jdx]

        if 4 <= i2s_prob.shape[0] <= 6:
            recipe_name = raw_info["name"]
            plot_attention_heatmap(i2s_prob, "images/{}_i2s.png".format(recipe_name))
            plot_attention_heatmap(s2s_prob, "images/{}_s2s.png".format(recipe_name))
