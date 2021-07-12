"""
This script handles the training process.
"""

import argparse
import math
import time
import pickle

import random
import numpy as np
import os
import json
import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.rtransformer.recursive_caption_dataset import \
    caption_collate, single_sentence_collate, prepare_batch_inputs
from src.rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset
from src.rtransformer.model import RecursiveTransformer, NonRecurTransformer, NonRecurTransformerUntied, TransformerXL, StructureAwareRecursiveTransformer
from src.rtransformer.masked_transformer import MTransformer
from src.rtransformer.optimization import BertAdam, EMA
from src.translator import Translator
from src.translate import run_translate
from src.utils import save_parsed_args_to_json, save_json, load_json, \
    count_parameters, merge_dicts
from easydict import EasyDict as EDict
from tensorboardX import SummaryWriter
import logging
logger = logging.getLogger(__name__)

def eval_epoch(model, validation_data, device, opt):
    """The same setting as training, where ground-truth word x_{t-1}
    is used to predict next word x_{t}, not realistic for real inference"""
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    attention_dicts = []
    batch_raw_infos = []

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc="  Validation =>"):
            if opt.recurrent:
                # prepare data
                batched_data = [prepare_batch_inputs(step_data, device=device, non_blocking=opt.pin_memory)
                                for step_data in batch[0]]
                input_ids_list = [e["input_ids"] for e in batched_data]
                video_features_list = [e["video_feature"] for e in batched_data]
                input_masks_list = [e["input_mask"] for e in batched_data]
                token_type_ids_list = [e["token_type_ids"] for e in batched_data]
                input_labels_list = [e["input_labels"] for e in batched_data]

                # ingredients
                batch_step_num = batch[1]
                ingr_input_ids = torch.LongTensor([e["ingr_ids"] for e in batch[3]]).cuda()
                ingr_masks = torch.LongTensor([e["ingr_mask"] for e in batch[3]]).cuda()
                ingr_sep_masks = torch.LongTensor([e["ingr_sep_mask"] for e in batch[3]]).cuda()

                ingr_id_dict = [e["ingr_id_dict"] for e in batch[3]]
                extra_zeros =  [len(e["oov_word_dict"]) for e in batch[3]]

                loss, pred_scores_list, attention_dict = model(input_ids_list, video_features_list,
                                                               input_masks_list, token_type_ids_list, 
                                                               input_labels_list, ingr_input_ids,
                                                               ingr_masks, ingr_sep_masks, batch_step_num,
                                                               ingr_id_dict, extra_zeros,
                                                               predict=True)

                attention_dicts.append(attention_dict)
                batch_raw_infos.append(batch[2])

                if opt.ours:
                    batch_input_labels_list = []
                    for batch_idx in range(len(batch_step_num)):
                        step_num = batch_step_num[batch_idx]
                        input_labels = [x[batch_idx][opt.max_v_len:] for x in input_labels_list[:step_num]]
                        input_labels = torch.stack(input_labels)
                        batch_input_labels_list.append(input_labels)

            else:  # single sentence
                if opt.untied or opt.mtrans:
                    # prepare data
                    batched_data = prepare_batch_inputs(batch[0], device=device, non_blocking=opt.pin_memory)
                    video_feature = batched_data["video_feature"]
                    video_mask = batched_data["video_mask"]
                    text_ids = batched_data["text_ids"]
                    text_mask = batched_data["text_mask"]
                    text_labels = batched_data["text_labels"]

                    loss, pred_scores = model(video_feature, video_mask, text_ids, text_mask, text_labels)
                    pred_scores_list = [pred_scores]
                    batch_input_labels_list = [text_labels]
                else:
                    # prepare data
                    batched_data = prepare_batch_inputs(batch[0], device=device, non_blocking=opt.pin_memory)
                    input_ids = batched_data["input_ids"]
                    video_features = batched_data["video_feature"]
                    input_masks = batched_data["input_mask"]
                    token_type_ids = batched_data["token_type_ids"]
                    input_labels = batched_data["input_labels"]

                    loss, pred_scores = model(input_ids, video_features, input_masks, token_type_ids, input_labels)
                    pred_scores_list = [pred_scores]
                    input_labels_list = [input_labels]

    return attention_dicts, batch_raw_infos

def get_args():
    """parse and preprocess cmd line args"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dset_name", type=str, default="anet", choices=["anet", "yc2"],
                        help="Name of the dataset, will affect data loader, evaluation, etc")

    # model config
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, help="number of words in the vocabulary")
    parser.add_argument("--word_vec_size", type=int, default=300)
    parser.add_argument("--video_feature_size", type=int, default=3072, help="2048 appearance + 1024 flow")
    parser.add_argument("--max_v_len", type=int, default=100, help="max length of video feature")
    parser.add_argument("--max_i_len", type=int, default=70, help="max length of ingredients")
    parser.add_argument("--max_t_len", type=int, default=25,
                        help="max length of text (sentence or paragraph), 30 for anet, 20 for yc2")
    parser.add_argument("--max_n_sen", type=int, default=6,
                        help="for recurrent, max number of sentences, 6 for anet, 10 for yc2")
    parser.add_argument("--n_memory_cells", type=int, default=1, help="number of memory cells in each layer")
    parser.add_argument("--type_vocab_size", type=int, default=2, help="video as 0, text as 1")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of transformer layers")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--memory_dropout_prob", type=float, default=0.1)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--glove_path", type=str, default=None, help="extracted GloVe vectors")
    parser.add_argument("--freeze_glove", action="store_true", help="do not train GloVe vectors")
    parser.add_argument("--share_wd_cls_weight", action="store_true",
                        help="share weight matrix of the word embedding with the final classifier, ")

    parser.add_argument("--recurrent", action="store_true", help="Run recurrent model")
    parser.add_argument("--untied", action="store_true", help="Run untied model")

    parser.add_argument("--ours", action="store_true", help="Use our algorithm")
    parser.add_argument("--full", action="store_true", help="use full model")
    parser.add_argument("--wo_refinements", action="store_true", help="w/o refinements")
    parser.add_argument("--copy", action="store_true", help="w/ copy only")
    parser.add_argument("--struct", action="store_true", help="w/ only struct")
    parser.add_argument("--ingr", action="store_true", help="w/ only ingredients")
    parser.add_argument("--video", action="store_true", help="w/ only video")

    parser.add_argument("--xl", action="store_true", help="transformer xl model, when specified, "
                                                          "will automatically set recurrent = True, "
                                                          "since the data loading part is the same")
    parser.add_argument("--xl_grad", action="store_true",
                        help="enable back-propagation for xl model, only useful when `-xl` flag is enabled."
                             "Note, the original transformerXL model does not allow back-propagation.")
    parser.add_argument("--mtrans", action="store_true",
                        help="Masked transformer model for single sentence generation")

    # training config -- learning rate
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--grad_clip", type=float, default=1, help="clip gradient, -1 == disable")
    parser.add_argument("--ema_decay", default=0.9999, type=float,
                        help="Use exponential moving average at training, float in (0, 1) and -1: do not use.  "
                             "ema_param = new_param * ema_decay + (1-ema_decay) * last_param")

    parser.add_argument("--data_dir", required=True, help="dir containing the splits data files")
    parser.add_argument("--video_feature_dir", required=True, help="dir containing the video features")
    parser.add_argument("--v_duration_file", required=True, help="filepath to the duration file")
    parser.add_argument("--word2idx_path", type=str, default="./cache/word2idx.json")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Use soft target instead of one-hot hard target")
    parser.add_argument("--n_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--max_es_cnt", type=int, default=10,
                        help="stop if the model is not improving for max_es_cnt max_es_cnt")
    parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
    parser.add_argument("--val_batch_size", type=int, default=50, help="inference batch size")

    parser.add_argument("--use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("--beam_size", type=int, default=2, help="beam size")
    parser.add_argument("--n_best", type=int, default=1, help="stop searching when get n_best from beam search")

    # others
    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Don't use pin_memory=True for dataloader. "
                             "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")
    parser.add_argument("---num_workers", type=int, default=0,
                        help="num subprocesses used to load the data, 0: use main process")
    parser.add_argument("--exp_id", type=str, default="res", help="id of the current run")
    parser.add_argument("--res_root_dir", type=str, default="results", help="dir to containing all the results")
    parser.add_argument("--save_model", default="model")
    parser.add_argument("--save_mode", type=str, choices=["all", "best"], default="best",
                        help="all: save models at each epoch; best: only save the best model")
    parser.add_argument("--no_cuda", action="store_true", help="run on cpu")
    parser.add_argument("--seed", default=2019, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval_tool_dir", type=str, default="./densevid_eval")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.recurrent = True if opt.xl else opt.recurrent
    assert not (opt.recurrent and opt.untied), "cannot be True for both"
    assert not (opt.recurrent and opt.mtrans), "cannot be True for both"
    assert not (opt.untied and opt.mtrans), "cannot be True for both"
    if opt.xl_grad:
        assert opt.xl, "`-xl` flag must be set when using `-xl_grad`."

    if opt.recurrent:  # recurrent + xl
        if opt.xl:
            model_type = "xl_grad" if opt.xl_grad else "xl"
        else:
            model_type = "re"
    else:  # single sentence
        if opt.untied:
            model_type = "untied_single"
        elif opt.mtrans:
            model_type = "mtrans_single"
        else:
            model_type = "single"

    # make paths
    opt.res_dir = os.path.join(
        opt.res_root_dir, "_".join([opt.dset_name, model_type, opt.exp_id, time.strftime("%Y_%m_%d_%H_%M_%S")]))
    if opt.debug:
        opt.res_dir = "debug_" + opt.res_dir

    if os.path.exists(opt.res_dir) and os.listdir(opt.res_dir):
        raise ValueError("File exists {}".format(opt.res_dir))
    elif not os.path.exists(opt.res_dir):
        os.makedirs(opt.res_dir)

    opt.log = os.path.join(opt.res_dir, opt.save_model)
    opt.save_model = os.path.join(opt.res_dir, opt.save_model)
    opt.pin_memory = not opt.no_pin_memory

    if opt.share_wd_cls_weight:
        assert opt.word_vec_size == opt.hidden_size, \
            "hidden size has to be the same as word embedding size when " \
            "sharing the word embedding weight and the final classifier weight"
    return opt


def main():
    opt = get_args()

    if opt.full:
        model_mode = "full"
    elif opt.wo_refinements:
        model_mode = "wo_refinements"
    elif opt.copy:
        model_mode = "copy"
    elif opt.struct:
        model_mode = "struct"
    elif opt.ingr:
        model_mode = "ingr"
    else:
        model_mode = "video"

    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    train_dataset = RCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, video_feature_dir="/mnt/LSTA5/data/common/recipe/youcook2/features/training",
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path, max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, max_n_sen=opt.max_n_sen, max_i_len=opt.max_i_len,
        mode="train", recurrent=opt.recurrent, untied=opt.untied or opt.mtrans)
    # add 10 at max_n_sen to make the inference stage use all the segments
    val_dataset = RCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, video_feature_dir="/mnt/LSTA5/data/common/recipe/youcook2/features/validation",
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path, max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, max_n_sen=opt.max_n_sen+10, max_i_len=opt.max_i_len,
        mode="val", recurrent=opt.recurrent, untied=opt.untied or opt.mtrans)

    if opt.recurrent or opt.ours:
        collate_fn = caption_collate
    else:  # single sentence (including untied)
        collate_fn = single_sentence_collate

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn,
                              batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn,
                            batch_size=opt.val_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=opt.pin_memory)

    opt.vocab_size = len(train_dataset.word2idx)
    print(json.dumps(vars(opt), indent=4, sort_keys=True))

    device = torch.device("cuda" if opt.cuda else "cpu")
    rt_config = EDict(
        xl_grad=opt.xl_grad,  # enable back-propagation for transformerXL model
        hidden_size=opt.hidden_size,
        intermediate_size=opt.intermediate_size,  # after each self attention
        vocab_size=opt.vocab_size,  # get from word2idx
        word_vec_size=opt.word_vec_size,
        video_feature_size=opt.video_feature_size,
        max_position_embeddings=opt.max_v_len + opt.max_t_len,  # get from max_seq_len
        max_v_len=opt.max_v_len,  # max length of the videos
        max_t_len=opt.max_t_len,  # max length of the text
        max_i_len=opt.max_i_len,  # max length of the ingredients
        model_mode=model_mode,    # for ablation study
        type_vocab_size=opt.type_vocab_size,
        layer_norm_eps=opt.layer_norm_eps,  # bert layernorm
        hidden_dropout_prob=opt.hidden_dropout_prob,  # applies everywhere except attention
        num_hidden_layers=opt.num_hidden_layers,  # number of transformer layers
        num_attention_heads=opt.num_attention_heads,
        attention_probs_dropout_prob=opt.attention_probs_dropout_prob,  # applies only to self attention
        n_memory_cells=opt.n_memory_cells,  # memory size will be (n_memory_cells, D)
        memory_dropout_prob=opt.memory_dropout_prob,
        initializer_range=opt.initializer_range,
        label_smoothing=opt.label_smoothing,
        share_wd_cls_weight=opt.share_wd_cls_weight
    )
    if opt.recurrent:
        if opt.ours:
            logger.info("Use step-denendency model - Ours")
            model = StructureAwareRecursiveTransformer(rt_config)
        elif opt.xl:
            logger.info("Use recurrent model - TransformerXL" + " (with gradient)" if opt.xl_grad else "")
            model = TransformerXL(rt_config)
        else:
            logger.info("Use recurrent model - Mine")
            model = RecursiveTransformer(rt_config)
    else:  # single sentence, including untied
        if opt.untied:
            logger.info("Use untied non-recurrent single sentence model")
            model = NonRecurTransformerUntied(rt_config)
        elif opt.mtrans:
            logger.info("Use masked transformer -- another non-recurrent single sentence model")
            model = MTransformer(rt_config)
        else:
            logger.info("Use non-recurrent single sentence model")
            model = NonRecurTransformer(rt_config)

    if opt.glove_path is not None:
        if hasattr(model, "ingredient_embeddings") and hasattr(model, "text_embeddings"):
            logger.info("Load GloVe as ingredient and word embedding")
            model.ingredient_embeddings.set_pretrained_embedding(
                torch.from_numpy(torch.load(opt.glove_path)).float(), freeze=opt.freeze_glove)
            model.text_embeddings.set_pretrained_embedding(
                torch.from_numpy(torch.load(opt.glove_path)).float(), freeze=opt.freeze_glove)
        elif hasattr(model, "embeddings"):
            logger.info("Load GloVe as word embedding")
            model.embeddings.set_pretrained_embedding(
                torch.from_numpy(torch.load(opt.glove_path)).float(), freeze=opt.freeze_glove)
        else:
            logger.warning("This model has no embeddings, cannot load glove vectors into the model")

    print("Loading checkpoint...")
    checkpoint = torch.load("/mnt/LSTA5/data/nishimura/graph_youcook2_generator/baselines/recurrent-transformer/features/model/wo_ingredients/full.chkpt")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print("Loaded")
    
    attention_dicts, batch_raw_infos = eval_epoch(model, val_loader, device, opt)
    output = {
            "attention_dicts" : attention_dicts,
            "batch_raw_infos" : batch_raw_infos
            }
    with open("/mnt/LSTA5/data/nishimura/graph_youcook2_generator/baselines/recurrent-transformer/features/model/wo_ingredients/output.pkl", "wb") as f:
        pickle.dump(output, f)

if __name__ == "__main__":
    main()
