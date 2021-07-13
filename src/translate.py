""" Translate input text with trained model. """

import os
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import random
import numpy as np
import subprocess
from collections import defaultdict

from src.translator import Translator
from src.rtransformer.recursive_caption_dataset import \
    caption_collate, single_sentence_collate, prepare_batch_inputs
from src.rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset
from src.utils import load_json, merge_dicts, save_json


def sort_res(res_dict):
    """res_dict: the submission json entry `results`"""
    final_res_dict = {}
    for k, v in res_dict.items():
        final_res_dict[k] = sorted(v, key=lambda x: float(x["timestamp"][0]))
    return final_res_dict

def remove_dup(words):
    """
    remove duplicated words
    """
    words = words.split(" ")
    prev_word = words[0]
    sentence = [prev_word]

    for w_idx in range(1, len(words)):
        cur_word = words[w_idx]
        if cur_word == prev_word:
            continue
        else:
            sentence.append(cur_word)
            prev_word = cur_word
    return " ".join(sentence)


def run_translate(eval_data_loader, translator, opt):
    # submission template
    batch_res = {"version": "VERSION 1.0",
                 "results": defaultdict(list),
                 "external_data": {"used": "true", "details": "ay"}}
    for raw_batch in tqdm(eval_data_loader, mininterval=2, desc="  - (Translate)"):
        # prepare data
        step_sizes = raw_batch[1]  # list(int), len == bsz
        meta = raw_batch[2]  # list(dict), len == bsz
        batch = [prepare_batch_inputs(step_data, device=translator.device)
                 for step_data in raw_batch[0]]
        
        model_inputs = [
            [e["input_ids"] for e in batch],
            [e["video_feature"] for e in batch],
            [e["input_mask"] for e in batch],
            [e["token_type_ids"] for e in batch],
            [e["ingr_ids"] for e in raw_batch[3]],
            [e["ingr_mask"] for e in raw_batch[3]],
            [e["ingr_sep_mask"] for e in raw_batch[3]],
            [e["ingr_id_dict"] for e in raw_batch[3]],
            [e["oov_word_dict"] for e in raw_batch[3]],
            [e.cuda() for e in raw_batch[4]],
            [e.cuda() for e in raw_batch[5]],
            step_sizes
        ]


        dec_seq_list, oov_word_dict = translator.translate_batch(model_inputs, 
                                                                 use_beam=opt.use_beam, 
                                                                 recurrent=False, 
                                                                 untied=False, xl=opt.xl)

        # example_idx indicates which example is in the batch
        for example_idx, (step_size, cur_meta) in enumerate(zip(step_sizes, meta)):
            # step_idx or we can also call it sen_idx
            for step_idx, step_batch in enumerate(dec_seq_list[example_idx]):
                sentence = eval_data_loader.dataset.convert_ids_to_sentence(step_batch.cpu().tolist(), oov_word_dict[example_idx])
                sentence = remove_dup(sentence)
                sentence = sentence.encode("ascii", "ignore")

                batch_res["results"][cur_meta["name"]].append({
                    "sentence": sentence,
                    "timestamp": cur_meta["timestamp"][step_idx],
                    "gt_sentence": cur_meta["gt_sentence"][step_idx]
                })

    batch_res["results"] = sort_res(batch_res["results"])
    return batch_res

def get_data_loader(opt, eval_mode="val"):
    eval_dataset = RCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, video_feature_dir=opt.video_feature_dir,
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path, max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, max_n_sen=opt.max_n_sen + 10, mode=eval_mode,
        recurrent=opt.recurrent, untied=opt.untied or opt.mtrans)

    if opt.recurrent:  # recurrent model
        collate_fn = caption_collate
    else:  # single sentence
        collate_fn = single_sentence_collate
    eval_data_loader = DataLoader(eval_dataset, collate_fn=collate_fn,
                                  batch_size=opt.batch_size, shuffle=False, num_workers=8)
    return eval_data_loader
