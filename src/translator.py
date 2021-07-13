""" This module will handle the text generation with beam search. """

import torch
import copy
import torch.nn.functional as F

from src.rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset

import logging
logger = logging.getLogger(__name__)


def mask_tokens_after_eos(input_ids, input_masks,
                          eos_token_id=RCDataset.EOS, pad_token_id=RCDataset.PAD):
    """replace values after `[EOS]` with `[PAD]`,
    used to compute memory for next sentence generation"""
    for row_idx in range(len(input_ids)):
        # possibly more than one `[EOS]`
        cur_eos_idxs = (input_ids[row_idx] == eos_token_id).nonzero()
        if len(cur_eos_idxs) != 0:
            cur_eos_idx = cur_eos_idxs[0, 0].item()
            input_ids[row_idx, cur_eos_idx+1:] = pad_token_id
            input_masks[row_idx, cur_eos_idx+1:] = 0
    return input_ids, input_masks


class Translator(object):
    """Load with trained model and handle the beam search"""
    def __init__(self, opt, checkpoint, model=None):
        self.opt = opt
        self.device = torch.device("cuda" if opt.cuda else "cpu")

        self.model_config = checkpoint["model_cfg"]
        self.max_t_len = self.model_config.max_t_len
        self.max_v_len = self.model_config.max_v_len
        self.num_hidden_layers = self.model_config.num_hidden_layers

        model.load_state_dict(checkpoint["model"])
        print("[Info] Trained model state loaded.")
        self.model = model
        self.model.eval()

        # self.eval_dataset = eval_dataset

    def translate_batch_greedy(self, input_ids_list, 
                               video_features_list, input_masks_list, token_type_ids_list,
                               ingr_input_ids, ingr_masks, ingr_sep_masks, ingr_id_dict, oov_word_dict,
                               alignments, actions, batch_step_num, rt_model):

        def greedy_decoding_step(input_ids, video_features, input_masks, token_type_ids,
                                 ingr_ids, ingr_mask, ingr_sep_mask, model, alignment, action,
                                 ingr_id_dic, oov_word_dic, max_v_len, max_t_len, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):

            step_num = len(input_ids)
            next_symbols = torch.LongTensor([start_idx] * step_num)  # (N, )

            ingr_vectors = model.ingredient_embeddings(ingr_ids.unsqueeze(0), ingr_sep_mask.unsqueeze(0))
            ga_ingr_vectors = ingr_vectors.squeeze(0)

            input_ids = torch.stack(input_ids)
            video_features = torch.stack(video_features)
            input_masks = torch.stack(input_masks)
            token_type_ids = torch.stack(token_type_ids)

            encoder_outputs = model.forward_step(input_ids, video_features, input_masks)
            encoder_outputs = model.step_positional_encoding(encoder_outputs[:, 0, :].unsqueeze(0))
            encoder_outputs = model.step_wise_encoder(encoder_outputs, torch.ones(1, step_num).cuda())[-1]
            ga_step_vectors = encoder_outputs.squeeze(0)

            # generate a recipe
            text_input_ids = input_ids[:, max_v_len:] 
            extended_text_input_ids = text_input_ids.detach().clone()
            text_masks = input_masks[:, max_v_len:]

            if model.config.model_mode == "full" or model.config.model_mode == "reason_copy": # VIVT or VIV
                ga_step_vectors = ga_step_vectors.unsqueeze(0)
                _, _, entity_vectors, step_entity_vectors, action_vectors = model.reasoner(ga_step_vectors, ga_ingr_vectors)
                entity_vectors = model.Went(entity_vectors)
                action_vectors = model.Wac(action_vectors)

                ga_step_vectors = ga_step_vectors.transpose(1, 0).contiguous()
                entity_vectors = entity_vectors.unsqueeze(1)
                action_vectors = action_vectors.unsqueeze(1)

                ga_inputs = torch.cat([ga_step_vectors, entity_vectors, action_vectors], dim=1)
                ga_mask = torch.ones((ga_inputs.shape[0], ga_inputs.shape[1])).cuda()

                for dec_idx in range(max_t_len):
                    if dec_idx == 0:
                        text_input_ids[:, dec_idx] = next_symbols
                        extended_text_input_ids[:, dec_idx] = next_symbols
                    else:
                        text_input_ids[:, dec_idx] = next_symbols
                        extended_text_input_ids[:, dec_idx] = oov_next_symbols

                    text_masks[:, dec_idx] = 1

                    text_embeddings = model.text_embeddings(text_input_ids)
                    decoder_outputs = model.decoder(text_embeddings, text_masks,
                                                    ga_inputs, ga_mask, diagonal_mask=True)[-1]

                    pred_scores = model.pointer_generator_network(decoder_outputs,
                                                                  step_entity_vectors,
                                                                  ingr_id_dic, len(oov_word_dic))

                    pred_scores[:, :, unk_idx] = -1e10
                    next_words = pred_scores[:, dec_idx].max(1)[1]
                    oov_next_symbols = next_words.detach().clone()

                    copied_word_idx = next_words >= pred_scores.shape[-1] - len(oov_word_dic)
                    next_words[copied_word_idx] = unk_idx # because text_embedding does not have it
                    next_symbols = next_words
                return extended_text_input_ids

            elif model.config.model_mode == "copy": # VI
                ga_step_vectors = ga_step_vectors.unsqueeze(1)
                ingr_ga_verbose_vectors = ga_ingr_vectors.clone().unsqueeze(0).repeat(step_num, 1, 1)
                step_ga_ingr_vectors = ga_ingr_vectors.mean(dim=0).unsqueeze(0).unsqueeze(0).repeat(ga_step_vectors.shape[0], 1, 1)
                ga_inputs = torch.cat([ga_step_vectors, step_ga_ingr_vectors], dim=1)
                ga_mask = torch.ones((ga_inputs.shape[0], ga_inputs.shape[1])).cuda()

                for dec_idx in range(max_t_len):
                    if dec_idx == 0:
                        text_input_ids[:, dec_idx] = next_symbols
                        extended_text_input_ids[:, dec_idx] = next_symbols
                    else:
                        text_input_ids[:, dec_idx] = next_symbols
                        extended_text_input_ids[:, dec_idx] = oov_next_symbols

                    text_masks[:, dec_idx] = 1

                    text_embeddings = model.text_embeddings(text_input_ids)
                    decoder_outputs = model.decoder(text_embeddings, text_masks,
                                                    ga_inputs, ga_mask, diagonal_mask=True)[-1]

                    pred_scores = model.pointer_generator_network(decoder_outputs,
                                                                  ingr_ga_verbose_vectors,
                                                                  ingr_id_dic, len(oov_word_dic))
                    pred_scores[:, :, unk_idx] = -1e10
                    next_words = pred_scores[:, dec_idx].max(1)[1]
                    oov_next_symbols = next_words.detach().clone()

                    copied_word_idx = next_words >= pred_scores.shape[-1] - len(oov_word_dic)
                    next_words[copied_word_idx] = unk_idx # because text_embedding does not have it
                    next_symbols = next_words
                return extended_text_input_ids

            else: # V
                ga_inputs = ga_step_vectors.unsqueeze(1)
                ga_mask = torch.ones((ga_inputs.shape[0], ga_inputs.shape[1])).cuda()

                for dec_idx in range(max_t_len):
                    text_input_ids[:, dec_idx] = next_symbols
                    text_masks[:, dec_idx] = 1

                    text_embeddings = model.text_embeddings(text_input_ids)
                    decoder_outputs = model.decoder(text_embeddings, text_masks, 
                                                   ga_inputs, ga_mask, diagonal_mask=True)[-1]
                    pred_scores = model.decoder_classifier(decoder_outputs)
                    pred_scores[:, :, unk_idx] = -1e10
                    next_words = pred_scores[:, dec_idx].max(1)[1]
                    next_symbols = next_words
                return text_input_ids

        input_ids_list, input_masks_list = self.prepare_video_only_inputs(
            input_ids_list, input_masks_list, token_type_ids_list)
        for cur_input_masks in input_ids_list:
            assert torch.sum(cur_input_masks[:, self.max_v_len + 1:]) == 0, \
                "Initially, all text tokens should be masked"

        config = rt_model.config
        dec_seq_list = []
        # recipe-level deq_seq (N * seq)
        with torch.no_grad():
            for batch_idx in range(len(batch_step_num)):
                step_num = batch_step_num[batch_idx]
                input_ids = [x[batch_idx] for x in input_ids_list[:step_num]]
                video_features = [x[batch_idx] for x in video_features_list[:step_num]]
                input_masks = [x[batch_idx] for x in input_masks_list[:step_num]]
                token_type_ids = [x[batch_idx] for x in token_type_ids_list[:step_num]]
                ingr_ids = torch.LongTensor(ingr_input_ids[batch_idx]).cuda()
                ingr_mask = torch.LongTensor(ingr_masks[batch_idx]).cuda()
                ingr_sep_mask = torch.LongTensor(ingr_sep_masks[batch_idx]).cuda()
                alignment, action = alignments[batch_idx], actions[batch_idx]
                ingr_id_dic = ingr_id_dict[batch_idx]
                oov_word_dic = oov_word_dict[batch_idx]

                dec_seq = greedy_decoding_step(input_ids, video_features, input_masks, token_type_ids,
                                               ingr_ids, ingr_masks, ingr_sep_mask, rt_model, 
                                               alignment, action, ingr_id_dic, oov_word_dic, config.max_v_len, config.max_t_len)
                dec_seq_list.append(dec_seq)
        return dec_seq_list, oov_word_dict

    def translate_batch(self, model_inputs, use_beam=False, recurrent=True, untied=False, xl=False, mtrans=False):
        """while we used *_list as the input names, they could be non-list for single sentence decoding case"""
        input_ids_list, video_features_list, input_masks_list, token_type_ids_list, \
            ingr_input_ids, ingr_masks, ingr_sep_masks, ingr_id_dict, oov_word_dict, \
                alignments, actions, batch_step_num = model_inputs

        return self.translate_batch_greedy(
                    input_ids_list, video_features_list, input_masks_list, 
                    token_type_ids_list, ingr_input_ids, ingr_masks, ingr_sep_masks, 
                    ingr_id_dict, oov_word_dict, alignments, actions, batch_step_num, self.model)

    @classmethod
    def prepare_video_only_inputs(cls, input_ids, input_masks, segment_ids):
        """ replace text_ids (except `[BOS]`) in input_ids with `[PAD]` token, for decoding.
        This function is essential!!!
        Args:
            input_ids: (N, L) or [(N, L)] * step_size
            input_masks: (N, L) or [(N, L)] * step_size
            segment_ids: (N, L) or [(N, L)] * step_size
        """
        if isinstance(input_ids, list):
            video_only_input_ids_list = []
            video_only_input_masks_list = []
            for e1, e2, e3 in zip(input_ids, input_masks, segment_ids):
                text_mask = e3 == 1  # text positions (`1`) are replaced
                e1[text_mask] = RCDataset.PAD
                e2[text_mask] = 0  # mark as invalid bits
                video_only_input_ids_list.append(e1)
                video_only_input_masks_list.append(e2)
            return video_only_input_ids_list, video_only_input_masks_list
        else:
            text_mask = segment_ids == 1
            input_ids[text_mask] = RCDataset.PAD
            input_masks[text_mask] = 0
            return input_ids, input_masks
