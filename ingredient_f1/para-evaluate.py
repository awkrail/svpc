# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# --------------------------------------------------------

import argparse
import json
import sys
import os
import pickle
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(file_dir, 'coco-caption')) # Hack to allow the import of pycocoeval

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
#from pycocoevalcap.re.re import Re
#from pycocoevalcap.self_bleu.self_bleu import Self_Bleu
#from sets import Set
import numpy as np
import re
import random

def parse_sent(sent):
    res = re.sub('[^a-zA-Z]', ' ', sent)
    res = res.strip().lower().split()
    return res

def parse_para(para):
    para = para.replace('..', '.')
    para = para.replace('.', ' endofsent')
    return parse_sent(para)

class ANETcaptions(object):

    def __init__(self, ground_truth_filenames=None, prediction_filename=None,
                 verbose=False, all_scorer=False):
        # Check that the gt and submission files exist and load them
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        self.verbose = verbose
        self.all_scorer = all_scorer
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.tokenizer = PTBTokenizer()

        # Set up scorers, if not verbose, we only use the one we're
        # testing on: METEOR
        if self.verbose or self.all_scorer:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                #(Meteor(),"METEOR"),
                #(Rouge(), "ROUGE_L"),
                #(Cider(), "CIDEr")
            ]
        else:
            self.scorers = [(Meteor(), "METEOR")]

    def ensure_caption_key(self, data):
        if len(data) == 0:
            return data
        if not list(data.keys())[0].startswith('v_'):
            data = {'v_' + k: data[k] for k in data}
        return data

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print("| Loading submission... {}".format(prediction_filename))
        submission = json.load(open(prediction_filename))['results']
        # change to paragraph format
        para_submission = {}
        for id in submission.keys():
            para_submission[id] = ''
            for info in submission[id]:
                para_submission[id] += info['sentence'] + '. '
        for para in para_submission.values():
            assert(type(para) == str or type(para) == unicode)
        # Ensure that every video is limited to the correct maximum number of proposals.
        return self.ensure_caption_key(para_submission)

    def import_ground_truths(self, filenames):
        gts = []
        self.n_ref_vids = set()
        for filename in filenames:
            gt = json.load(open(filename))
            self.n_ref_vids.update(gt.keys())
            gts.append(self.ensure_caption_key(gt))
        if self.verbose:
            print("| Loading GT. #files: %d, #videos: %d" % (len(filenames), len(self.n_ref_vids)))
        return gts

    def check_gt_exists(self, vid_id):
        for gt in self.ground_truths:
            if vid_id in gt:
              return True
        return False

    def get_gt_vid_ids(self):
        vid_ids = set([])
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate(self):
        recipe_ids = self.evaluate_para()
        random.shuffle(recipe_ids)
        return recipe_ids

    def evaluate_para(self):
        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos 
        gt_vid_ids = self.get_gt_vid_ids()
        vid2idx = {k: i for i, k in enumerate(gt_vid_ids)}
        gts = {vid2idx[k]: [] for k in gt_vid_ids}
        for i, gt in enumerate(self.ground_truths):
            for k in gt_vid_ids:
                if k not in gt:
                    continue
                gts[vid2idx[k]].append(' '.join(parse_sent(gt[k])))
        res = {vid2idx[k]: [' '.join(parse_sent(self.prediction[k]))] \
            if k in self.prediction and len(self.prediction[k]) > 0 else [''] for k in gt_vid_ids}
        para_res = {vid2idx[k]: [' '.join(parse_para(self.prediction[k]))] \
            if k in self.prediction and len(self.prediction[k]) > 0 else [''] for k in gt_vid_ids}

        # Each scorer will compute across all videos and take average score
        output = {}
        num = len(res)
        hard_samples = {}
        easy_samples = {}
        for scorer, method in self.scorers:
            if self.verbose:
                print('computing %s score...'%(scorer.method()))

            if method != 'Self_Bleu':
                score, scores = scorer.compute_score(gts, res)
            else:
                score, scores = scorer.compute_score(gts, para_res)
            scores = np.asarray(scores)
        print('# scored video =', num)
        top_120_recipes = scores[-1].argsort()[::-1][:120]
        top_120_recipe_ids = [gt_vid_ids[recipe_id][2:] for recipe_id in top_120_recipes]
        return top_120_recipe_ids

def load(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def main(args):
    # Call coco eval
    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             verbose=args.verbose,
                             all_scorer=args.all_scorer)
    recipe_ids = evaluator.evaluate()
    output = {}

    mart_w_ing_path = "eval_json/mart_w_ing.json"
    reason_copy_path = "eval_json/reason_copy.json"
    full_path = "eval_json/full.json"
    test_annotation_path = "../../preprocess/split/bosselut_split_yc2_test_anet_format.json"

    mart_recipes = load(mart_w_ing_path)
    reason_recipes = load(reason_copy_path)
    repred_recipes = load(full_path)
    test_annotation = load(test_annotation_path)

    result_dicts = []

    for recipe_id in recipe_ids:
        mart_recipe = [x["sentence"] for x in mart_recipes["results"][recipe_id]]
        reason_recipe = [x["sentence"] for x in reason_recipes["results"][recipe_id]]
        repred_recipe = [x["sentence"] for x in repred_recipes["results"][recipe_id]]
        gt_recipe = test_annotation[recipe_id]["sentences"]
        ingredients = test_annotation[recipe_id]["ingredients"]
        result_dicts.append({
            "ingredients" : ingredients,
            "recipe_id" : recipe_id,
            "recipes" : {
                "baseline" : mart_recipe,
                "half" : reason_recipe,
                "full" : repred_recipe,
                "gt" : gt_recipe
                }
            })
    
    with open("eval_json/human_evaluation.pkl", "wb") as f:
        pickle.dump(result_dicts, f)

import time
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str,  default='sample_submission.json',
                        help='sample submission file for ActivityNet Captions Challenge.')
    parser.add_argument('-r', '--references', type=str, nargs='+', required=True,
                        help='reference files with ground truth captions to compare results against. delimited (,) str')
    parser.add_argument('-o', '--output', type=str, default=None, help='output file with final language metrics.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print intermediate steps.')
    parser.add_argument('--time', '--t', action = 'store_true',
                        help = 'Count running time.')
    parser.add_argument('--all_scorer', '--a', action = 'store_true',
                        help = 'Use all scorer.')
    args = parser.parse_args()

    if args.output is None:
        r_path = args.submission
        r_path_splits = r_path.split(".")
        r_path_splits = r_path_splits[:-1] + ["_metric", r_path_splits[-1]]
        args.output = ".".join(r_path_splits)

    if args.time:
        start_time = time.time()
    main(args)
    if args.time:
        print('time = %.2f' % (time.time() - start_time))
