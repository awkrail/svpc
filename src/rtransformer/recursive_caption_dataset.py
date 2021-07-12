import copy
import torch
import logging
import math
import nltk
import numpy as np
import os

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.utils import load_json, flat_list_of_lists

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


class RecursiveCaptionDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"
    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    IGNORE = -1  # used to calculate loss

    """
    recurrent: if True, return recurrent data
    """
    def __init__(self, dset_name, data_dir, video_feature_dir, duration_file, word2idx_path,
                 verb_word2idx_path, max_t_len, max_v_len, max_i_len, max_n_sen, mode="train", 
                 recurrent=True, untied=False):
        self.dset_name = dset_name
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.verb2idx = load_json(verb_word2idx_path)
        self.action_vocab_size = len(self.verb2idx)
        self.data_dir = data_dir  # containing training data
        self.video_feature_dir = video_feature_dir  # a set of .h5 files
        self.duration_file = duration_file
        self.frame_to_second = self._load_duration()
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sen
        self.max_i_len = max_i_len
        self.max_n_sen = max_n_sen

        self.mode = mode
        self.recurrent = recurrent
        self.untied = untied
        assert not (self.recurrent and self.untied), "untied and recurrent cannot be True for both"

        # data entries
        self.data = None
        self.set_data_mode(mode=mode)
        self.missing_video_names = []
        self.fix_missing()

        self.num_sens = None  # number of sentence for each video, set in self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.recurrent:
            items, meta, ingr_items, alignment, actions = self.convert_example_to_features(self.data[index])
            return items, meta, ingr_items, alignment, actions
        else:
            items, meta = self.convert_example_to_features(self.data[index])
            return items, meta

    def set_data_mode(self, mode):
        """mode: `train` or `val`"""
        logging.info("Mode {}".format(mode))
        self.mode = mode
        if mode == "train":  # 10000 videos
            data_path = os.path.join(self.data_dir, "bosselut_yc2_train_anet_format.json")
        elif mode == "val":  # 2500 videos
            data_path = os.path.join(self.data_dir, "bosselut_split_yc2_val_anet_format.json")
        elif mode == "test":
            data_path = os.path.join(self.data_dir, "bosselut_split_yc2_test_anet_format.json")
        else:
            raise ValueError("Expecting mode to be one of [`train`, `val`, `test`], got {}".format(mode))
        self._load_data(data_path)

    def fix_missing(self):
        """filter our videos with no feature file"""
        for e in tqdm(self.data):
            video_name = e["name"][2:] if self.dset_name == "anet" else e["name"]
            cur_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
            cur_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
            for p in [cur_path_bn, cur_path_resnet]:
                if not os.path.exists(p):
                    self.missing_video_names.append(video_name)
        print("Missing {} features (clips/sentences) from {} videos".format(
            len(self.missing_video_names), len(set(self.missing_video_names))))
        print("Missing {}".format(set(self.missing_video_names)))
        if self.dset_name == "anet":
            self.data = [e for e in self.data if e["name"][2:] not in self.missing_video_names]
        else:
            self.data = [e for e in self.data if e["name"] not in self.missing_video_names]

    def _load_duration(self):
        """https://github.com/salesforce/densecap/blob/master/data/anet_dataset.py#L120
        Since the features are extracted not at the exact 0.5 secs. To get the real time for each feature,
        use `(idx + 1) * frame_to_second[vid_name] `
        """
        frame_to_second = {}
        sampling_sec = 0.5  # hard coded, only support 0.5
        if self.dset_name == "anet":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    frame_to_second[vid_name] = float(vid_dur) * int(
                        float(vid_frame) * 1. / int(float(vid_dur)) * sampling_sec) * 1. / float(vid_frame)
                frame_to_second["_0CqozZun3U"] = sampling_sec  # a missing video in anet
        elif self.dset_name == "yc2":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    frame_to_second[vid_name] = float(vid_dur) * math.ceil(
                        float(vid_frame) * 1. / float(vid_dur) * sampling_sec) * 1. / float(vid_frame)  # for yc2
        else:
            raise NotImplementedError("Only support anet and yc2, got {}".format(self.dset_name))
        return frame_to_second

    def _load_data(self, data_path):
        """
        {
            "duration" : xxx,
            "timestamps" : [[start, end], [start, end], ...,],
            "sentences" : ["sent1", "sent2", ...]
        }
        """
        logging.info("Loading data from {}".format(data_path))
        raw_data = load_json(data_path)
        data = []

        for k, line in tqdm(raw_data.items()):
            line["name"] = k
            line["timestamps"] = line["timestamps"][:self.max_n_sen]
            line["sentences"] = line["sentences"][:self.max_n_sen]
            data.append(line)

        if self.recurrent:  # recurrent
            self.data = data
        else:  # non-recurrent single sentence
            singel_sentence_data = []
            for d in data:
                num_sen = min(self.max_n_sen, len(d["sentences"]))
                singel_sentence_data.extend([
                    {
                        "duration": d["duration"],
                        "name": d["name"],
                        "timestamp": d["timestamps"][idx],
                        "sentence": d["sentences"][idx]
                    } for idx in range(num_sen)])
            self.data = singel_sentence_data

        logging.info("Loading complete! {} examples".format(len(self)))

    def convert_example_to_features(self, example):
        # 特徴ベクトルへ変換
        """example single snetence
        {"name": str,
         "duration": float,
         "timestamp": [st(float), ed(float)],
         "sentence": str
        } or
        {"name": str,
         "duration": float,
         "timestamps": list([st(float), ed(float)]),
         "sentences": list(str)
        }
        """
        name = example["name"]
        video_name = name[2:] if self.dset_name == "anet" else name
        feat_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
        feat_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
        video_feature = np.concatenate([np.load(feat_path_resnet), np.load(feat_path_bn)], axis=1)

        if self.recurrent:  # recurrent
            num_sen = len(example["sentences"])
            single_video_features = []
            single_video_meta = []
            ingr_data = self.clip_ingredient_to_feature(example["ingredients"])
            alignment = self.convert_alignment_to_onehot(example["alignment"], max_step_num=num_sen)
            actions = self.convert_action_to_onehot(example["action"])

            for clip_idx in range(num_sen):
                cur_data, cur_meta = self.clip_sentence_to_feature(example["name"],
                                                                   example["timestamps"][clip_idx],
                                                                   example["sentences"][clip_idx],
                                                                   ingr_data,
                                                                   video_feature)
                single_video_features.append(cur_data)
                single_video_meta.append(cur_meta)

            return single_video_features, single_video_meta, ingr_data, alignment, actions

        else:  # single sentence
            clip_dataloader = self.clip_sentence_to_feature_untied \
                if self.untied else self.clip_sentence_to_feature
            cur_data, cur_meta = clip_dataloader(example["name"],
                                                 example["timestamp"],
                                                 example["sentence"],
                                                 video_feature)
            return cur_data, cur_meta
    
    def convert_alignment_to_onehot(self, alignment, max_step_num):
        """
        for step_i in range(max_step_num):
            onehot = [0] * len(alignment) # #num_of_ingr
            for i, x in enumerate(alignment):
                if x < self.max_n_sen:
                    if x == step_i:
                        onehot[i] = 1
            one_hot_vector.append(onehot)
        """
        one_hot_vector = alignment[:self.max_n_sen]
        return torch.Tensor(one_hot_vector)

    def convert_action_to_onehot(self, actions):
        action_onehot = torch.zeros((len(actions), self.action_vocab_size))
        for i, step_action in enumerate(actions):
            for action, action_id in step_action.items():
                action_onehot[i, action_id] = 1.
        return action_onehot[:self.max_n_sen]

    def clip_ingredient_to_feature(self, ingredients):
        ingr_tokens, ingr_masks, ingr_sep_mask = self._tokenize_ingredients(ingredients)
        ingr_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in ingr_tokens]

        # for pointer-generator network
        vocab_size = len(self.word2idx)
        oov_num = 0
        oov_word_dict = {}
        ingr_extended_ids = []
        for i, ingr_id in enumerate(ingr_ids):
            if ingr_id == self.word2idx["[UNK]"]:
                if ingr_tokens[i] in oov_word_dict:
                    ingr_extended_ids.append(oov_word_dict[ingr_tokens[i]])
                else:
                    ingr_extended_ids.append(vocab_size + oov_num)
                    oov_word_dict[ingr_tokens[i]] = vocab_size + oov_num
                    oov_num += 1
            else:
                ingr_extended_ids.append(ingr_id)
        
        ingr_id_dict = {}
        ingr_id_num = 0
        for ingr_extended_id in ingr_extended_ids:
            if ingr_extended_id == 0:
                break # 0 == [PAD]
            if ingr_extended_id == 2:
                ingr_id_num += 1
            else:
                if ingr_id_num in ingr_id_dict:
                    ingr_id_dict[ingr_id_num].append(ingr_extended_id)
                else:
                    ingr_id_dict[ingr_id_num] = [ingr_extended_id]

        data = dict(
                ingr_tokens=ingr_tokens,
                ingr_ids=ingr_ids,
                ingr_extended_ids=ingr_extended_ids,
                oov_word_dict=oov_word_dict,
                ingr_id_dict=ingr_id_dict,
                ingr_mask=ingr_masks,
                ingr_sep_mask=ingr_sep_mask
                )
        return data
    
    def clip_sentence_to_feature(self, name, timestamp, sentence, ingr_data, video_feature):
        """ make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            video_feature: np array
        """
        frm2sec = self.frame_to_second[name[2:]] if self.dset_name == "anet" else self.frame_to_second[name]

        # video + text tokens
        feat, video_tokens, video_mask = self._load_indexed_video_feature(video_feature, timestamp, frm2sec)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        # 論文中のTE(token type embedding)のためのもの
        input_tokens = video_tokens + text_tokens

        # input_ids -> [[VIDEO..], [(Videoの)PAD..], [BOS], [words], [EOS], [PAD...]]
        input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens]

        extended_input_ids = []
        for input_token in input_tokens:
            if input_token in self.word2idx:
                extended_input_ids.append(self.word2idx[input_token])
            else:
                oov_ingr_word_dict = ingr_data["oov_word_dict"]
                if input_token in oov_ingr_word_dict:
                    extended_input_ids.append(oov_ingr_word_dict[input_token])
                else:
                    extended_input_ids.append(self.word2idx[self.UNK_TOKEN])
        
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        input_labels = \
            [self.IGNORE] * len(video_tokens) + \
            [self.IGNORE if m == 0 else tid for tid, m in zip(extended_input_ids[-len(text_mask):], text_mask)][1:] + \
            [self.IGNORE]

        input_mask = video_mask + text_mask
        token_type_ids = [0] * self.max_v_len + [1] * self.max_t_len

        data = dict(
            name=name,
            input_tokens=input_tokens,
            # model inputs
            input_ids=np.array(input_ids).astype(np.int64), # input_ids -> [[VIDEO..], [(Videoの)PAD..], [BOS], [words], [EOS], [PAD...]]
            input_labels=np.array(input_labels).astype(np.int64), # input_labels -> [-1, -1, ..., (単語のところのID), -1, -1...]
            input_mask=np.array(input_mask).astype(np.float32), # [videoのmaskされていないところ(=1 or 0(maskあり))] + [textのmaskされていないところ(=1 or 0)]
            token_type_ids=np.array(token_type_ids).astype(np.int64), # [videoの部分=0] + [textの部分=1]
            video_feature=feat.astype(np.float32) # segmentぶんの特徴ベクトル
        )
        meta = dict(
            # meta
            name=name,
            timestamp=timestamp,
            sentence=sentence,
        )
        return data, meta

    def clip_sentence_to_feature_untied(self, name, timestamp, sentence, raw_video_feature):
        """ make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            raw_video_feature: np array, N x D, for the whole video
        """
        frm2sec = self.frame_to_second[name[2:]] if self.dset_name == "anet" else self.frame_to_second[name]

        # video + text tokens
        video_feature, video_mask = self._load_indexed_video_feature_untied(raw_video_feature, timestamp, frm2sec)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        text_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in text_tokens]
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        text_labels = [self.IGNORE if m == 0 else tid for tid, m in zip(text_ids, text_mask)][1:] + [self.IGNORE]

        data = dict(
            name=name,
            text_tokens=text_tokens,
            # model inputs
            text_ids=np.array(text_ids).astype(np.int64),
            text_mask=np.array(text_mask).astype(np.float32),
            text_labels=np.array(text_labels).astype(np.int64),
            video_feature=video_feature.astype(np.float32),
            video_mask=np.array(video_mask).astype(np.float32),
        )
        meta = dict(
            # meta
            name=name,
            timestamp=timestamp,
            sentence=sentence,
        )
        return data, meta

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, frm2sec):
        """convert wall time st_ed to feature index st_ed"""
        st = int(math.floor(timestamp[0] / frm2sec))
        ed = int(math.ceil(timestamp[1] / frm2sec))
        ed = min(ed, feat_len-1)
        st = min(st, ed-1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed

    def _load_indexed_video_feature(self, raw_feat, timestamp, frm2sec):
        """ [CLS], [VID], ..., [VID], [SEP], [PAD], ..., [PAD],
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat is padded to length of (self.max_v_len + self.max_t_len,)
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """
        max_v_l = self.max_v_len - 2
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1

        feat = np.zeros((self.max_v_len + self.max_t_len, raw_feat.shape[1]))  # includes [CLS], [SEP]
        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat[1:max_v_l+1] = raw_feat[downsamlp_indices]  # truncate, sample???

            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            mask = [1] * (max_v_l + 2)
        else:
            valid_l = ed - st + 1 # valid_l -> videoの入っている分だけはVID_TOKENを入れる
            feat[1:valid_l+1] = raw_feat[st:ed + 1] ## ここでraw_feat(videoのベクトル)からst:edまでを抽出
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + \
                           [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)
            mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)
        return feat, video_tokens, mask

    def _load_indexed_video_feature_untied(self, raw_feat, timestamp, frm2sec):
        """ Untied version: [VID], ..., [VID], [PAD], ..., [PAD], len == max_v_len
        Returns:
            feat is padded to length of (self.max_v_len,)
            mask: self.max_v_len, with 1 indicates valid bits, 0 indicates padding
        """
        max_v_l = self.max_v_len
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1

        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat = raw_feat[downsamlp_indices]  # truncate, sample???
            mask = [1] * max_v_l  # no padding
        else:
            feat = np.zeros((max_v_l, raw_feat.shape[1]))  # only video features and padding
            valid_l = ed - st + 1
            feat[:valid_l] = raw_feat[st:ed + 1]
            mask = [1] * valid_l + [0] * (max_v_l - valid_l)
        return feat, mask
    
    def _tokenize_ingredients(self, ingredients):
        max_i_len = self.max_i_len
        ingredients = [nltk.tokenize.word_tokenize(ingredient.lower()) for ingredient in ingredients]
        ingredient_tokens = []
        for ingredient in ingredients:
            for word in ingredient:
                ingredient_tokens.append(word)
            ingredient_tokens.append(self.SEP_TOKEN)
        
        valid_l = len(ingredient_tokens)
        mask = [1] * valid_l + [0] * (max_i_len - valid_l)
        ingredient_tokens += [self.PAD_TOKEN] * (max_i_len - valid_l)
        ingredient_tokens = ingredient_tokens[:max_i_len]
        mask = mask[:max_i_len]
        ingredient_sep_tokens = [1 if token == "[SEP]" else 0 for token in ingredient_tokens]
        return ingredient_tokens, mask, ingredient_sep_tokens

    def _tokenize_pad_sentence(self, sentence):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_t_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, oov_word_dict, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        oov_idx2word = {v:k for k,v in oov_word_dict.items()}

        if rm_padding:
            #raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
            raw_words = []
            for wid in ids:
                if wid in self.idx2word:
                    if wid not in [self.PAD, self.IGNORE]:
                        raw_words.append(self.idx2word[wid])
                else:
                    raw_words.append(oov_idx2word[wid])

        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


def prepare_batch_inputs(batch, device, non_blocking=False):
    batch_inputs = dict()
    bsz = len(batch["name"])
    for k, v in batch.items():
        assert bsz == len(v), (bsz, k, v)
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        else:  # all non-tensor values
            batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            # ここでbatchぶん回収
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def caption_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66

    HOW to batch clip-sentence pair?
    1) directly copy the last sentence, but do not count them in when back-prop OR
    2) put all -1 to their text token label, treat
    """
    # ingredients
    raw_batch_ingr_data = [e[2] for e in batch]
    raw_batch_alignment = [e[3] for e in batch]
    raw_batch_actions = [e[4] for e in batch]

    # collect meta
    raw_batch_meta = [e[1] for e in batch]
    batch_meta = []
    for e in raw_batch_meta:
        cur_meta = dict(
            name=None,
            timestamp=[],
            gt_sentence=[]
        )
        for d in e:
            cur_meta["name"] = d["name"]
            cur_meta["timestamp"].append(d["timestamp"])
            cur_meta["gt_sentence"].append(d["sentence"])
        batch_meta.append(cur_meta)

    batch = [e[0] for e in batch]
    # Step1: pad each example to max_n_sen
    max_n_sen = max([len(e) for e in batch])
    raw_step_sizes = []

    padded_batch = []
    padding_clip_sen_data = copy.deepcopy(batch[0][0])  # doesn"t matter which one is used
    padding_clip_sen_data["input_labels"][:] = RecursiveCaptionDataset.IGNORE
    for ele in batch:
        cur_n_sen = len(ele)
        if cur_n_sen < max_n_sen:
            ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
        raw_step_sizes.append(cur_n_sen)
        padded_batch.append(ele)

    # Step2: batching each steps individually in the batches
    collated_step_batch = []
    for step_idx in range(max_n_sen):
        collated_step = step_collate([e[step_idx] for e in padded_batch])
        collated_step_batch.append(collated_step)
    return collated_step_batch, raw_step_sizes, batch_meta, raw_batch_ingr_data, raw_batch_alignment, raw_batch_actions


def single_sentence_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    # collect meta
    batch_meta = [{"name": e[1]["name"],
                   "timestamp": e[1]["timestamp"],
                   "gt_sentence": e[1]["sentence"]
                   } for e in batch]  # change key
    padded_batch = step_collate([e[0] for e in batch])
    return padded_batch, None, batch_meta
