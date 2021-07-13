import sys
import json
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import logging
from libs.ASL.src.loss_functions.losses import AsymmetricLoss as AsymmetricLoss

logger = logging.getLogger(__name__)


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        #self.log_softmax = nn.LogSoftmax(dim=-1)
        self.eps = 1e-12 # for numerical stablization

        # ここでエラー
        # smoothing_value = label_smoothing / (tgt_vocab_size - 1)  # count for the ground-truth word
        # one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        # self.register_buffer("one_hot", one_hot.unsqueeze(0))
        self.label_smoothing = label_smoothing
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in [-1, tgt_vocab_size-1], `-1` is ignored
        """
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        #output = self.log_softmax(output[valid_indices])
        output = torch.log(output[valid_indices] + self.eps)
        
        # Todo [木]: Label_smoothing_lossの実装
        tgt_vocab_size = output.shape[1]
        smoothing_value = self.label_smoothing / (tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value).cuda()
        one_hot[self.ignore_index] = 0

        model_prob = one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x

class IngredientPositionEncoding(nn.Module):
    def __init__(self, n_filters=128, max_len=500):
        super(IngredientPositionEncoding, self).__init__()
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x, ingr_sep_masks):
        """
        TODO: あっているか確認
        """
        batch_idx = len(x)
        max_ingr_num = (ingr_sep_masks == 1).sum(dim=1).max().item()
        ingr_dim = x.shape[-1]
        ingredient_vectors = torch.zeros((batch_idx, max_ingr_num, ingr_dim)).cuda()

        for b_idx in range(batch_idx):
            ingr_sep_mask = ingr_sep_masks[b_idx]
            sep_indices = (ingr_sep_mask == 1).nonzero().view(-1)
            prev_index = 0

            for s_jdx, sep_index in enumerate(sep_indices):
                ingredient_index = sep_index - prev_index + 1
                out = x[b_idx][prev_index:prev_index+ingredient_index] # self.pe[:ingredient_index]
                ingredient_vectors[b_idx, s_jdx] = out[:-1].mean(dim=0)
                prev_index = sep_index + 1

        # ingredient-level PE
        pe = self.pe.data[:max_ingr_num, :]
        pe = pe.unsqueeze(0)
        ingredient_vectors = ingredient_vectors + pe
        return ingredient_vectors


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertIntermediate3x(nn.Module):
    def __init__(self, config):
        super(BertIntermediate3x, self).__init__()
        self.dense = nn.Linear(config.intermediate_size*3, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


def make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """
    Args:
        input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
        max_v_len: int, the first `max_v_len` is for video and its padding, the length
            of the rest of the bits is `max_t_len`. We have L = `max_v_len` + `max_t_len`.
            Note max_v_len may also include the memory len (M), thus max_v_len += M
        max_t_len: int
        memory_len: int, M
    Returns:

    >>> max_v_len = 2; max_t_len=3; input_mask = torch.randn(2, 5)
    >>> make_pad_shifted_mask(input_mask, max_v_len, max_t_len)[0]
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])
    """
    bsz, seq_len = input_mask.shape
    assert max_v_len + max_t_len + memory_len == seq_len
    shifted_mask = input_mask.new_zeros(bsz, max_v_len + max_t_len, seq_len)  # (N, L, M+L)
    shifted_mask[:, :, :memory_len+max_v_len] = 1
    shifted_mask[:, max_v_len:, memory_len+max_v_len:] = \
        torch.tril(input_mask.new_ones(max_t_len, max_t_len), diagonal=0)
    return shifted_mask


def make_pad_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """input_mask: (N, L), """
    shifted_mask = make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=memory_len)
    # It's correct to use `input_mask.unsqueeze(1)' instead of
    # `torch.bmm(input_mask.unsqueeze(2), input_mask.unsqueeze(1))'
    # since the rest of the bits are still masked in the subsequent processing steps.
    pad_shifted_mask = shifted_mask * input_mask.unsqueeze(1)
    return pad_shifted_mask


def make_video_only_mask(input_mask, max_v_len):
    video_only_mask = copy.deepcopy(input_mask)
    video_only_mask[:, max_v_len:] = 0
    return video_only_mask


class BertLayerNoMemory(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemory, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """
        max_v_len, max_t_len = self.config.max_v_len, self.config.max_t_len
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertEncoderNoMemory(nn.Module):
    def __init__(self, config):
        super(BertEncoderNoMemory, self).__init__()
        self.layer = nn.ModuleList([BertLayerNoMemory(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertEmbeddingsWithVideo(nn.Module):
    """Construct the embeddings from word (+ video), position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    論文3.2のH^0 = [H^0_{video} ; H^0_{text}]に対応. [.]はconcat. -> H^0を返すクラス
    """
    def __init__(self, config, add_postion_embeddings=True):
        super(BertEmbeddingsWithVideo, self).__init__()
        """add_postion_embeddings: whether to add absolute positional embeddings"""
        self.add_postion_embeddings = add_postion_embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.video_embeddings = nn.Sequential(
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        
        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
                                                        max_len=config.max_position_embeddings)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """Note the from_pretrained does not work in-place, so you need to assign value to the embedding"""
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)

    def forward(self, input_ids, video_features, token_type_ids):
        """
        Args:
            input_ids: (N, L)
            video_features: (N, L, D)
            token_type_ids: (N, L, D)

        Returns:

        """
        words_embeddings = self.word_fc(self.word_embeddings(input_ids))
        video_embeddings = self.video_embeddings(video_features)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + video_embeddings + token_type_embeddings
        if self.add_postion_embeddings:
            embeddings = self.position_embeddings(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (N, L, D)


class BertEmbeddingsTextUntied(nn.Module):
    """Construct the embeddings from word and video, separately. position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    """
    def __init__(self, config):
        super(BertEmbeddingsTextUntied, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.position_embeddings_text = PositionEncoding(n_filters=config.hidden_size,
                                                         max_len=config.max_position_embeddings)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """Note the from_pretrained does not work in-place, so you need to assign value to the embedding"""
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)

    def forward(self, text_input_ids):
        """ text_input_ids: (N, Lt) """
        words_embeddings = self.word_fc(self.word_embeddings(text_input_ids))  # (N, Lt, D)
        words_embeddings = self.position_embeddings_text(words_embeddings)
        return words_embeddings  # (N, Lt, D)

class BertEmbeddingsIngredientsUntied(nn.Module):
    def __init__(self, config):
        super(BertEmbeddingsIngredientsUntied, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(
                BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.word_vec_size, config.hidden_size),
                nn.ReLU(True),
                BertLayerNorm(config.lstm_hidden_size, eps=config.layer_norm_eps) # necessary?
                )
        self.position_embeddings_ingr = IngredientPositionEncoding(n_filters=config.lstm_hidden_size,
                                                                   max_len=config.max_position_embeddings)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)
    
    def forward(self, ingr_input_ids, ingr_sep_masks):
        words_embeddings = self.word_fc(self.word_embeddings(ingr_input_ids))  # (N, Ling, D)
        words_embeddings = self.position_embeddings_ingr(words_embeddings, ingr_sep_masks) # (N, Ling, D)
        return words_embeddings


class BertEmbeddingsVideoUntied(nn.Module):
    """Construct the embeddings from word and video, separately. position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    """
    def __init__(self, config):
        super(BertEmbeddingsVideoUntied, self).__init__()
        self.video_embeddings = nn.Sequential(
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.position_embeddings_video = PositionEncoding(n_filters=config.hidden_size,
                                                          max_len=config.max_position_embeddings)

    def forward(self, video_features):
        """ video_features: (N, Lv, D) """
        video_embeddings = self.video_embeddings(video_features)  # (N, Lv, D)
        video_embeddings = self.position_embeddings_video(video_embeddings)
        return video_embeddings  # (N, Lv, D)


class BertLayerNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemoryUntied, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, diagonal_mask=False):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        Returns:

        """
        self_attention_mask = attention_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = hidden_states.size(1)
            self_attention_mask = self_attention_mask * \
                torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)
        attention_output = self.attention(hidden_states, self_attention_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertEncoderNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertEncoderNoMemoryUntied, self).__init__()
        self.layer = nn.ModuleList([BertLayerNoMemoryUntied(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, diagonal_mask=False, output_all_encoded_layers=True):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, diagonal_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertDecoderLayerNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertDecoderLayerNoMemoryUntied, self).__init__()
        self.config = config
        self.self_attention = BertSelfAttention(config)
        self.norm1 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dec_enc_attention = BertSelfAttention(config)
        self.norm2 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output = BertOutput(config)  # linear + residual + layernorm

    def forward(self, dec_hidden_states, dec_mask, enc_outputs, enc_mask, diagonal_mask=True):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        Returns:

        """
        self_attention_mask = dec_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = dec_mask.size(1)  # Lt
            self_attention_mask = self_attention_mask * \
                torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)

        # 1, dec self attn + add_norm
        attention_output = self.self_attention(
            dec_hidden_states, dec_hidden_states, dec_hidden_states, self_attention_mask)  # (N, Lt, D)
        attention_output = self.norm1(attention_output + dec_hidden_states)  # (N, Lt, D)

        # 2, dec enc attn + add_norm
        # Is the attention mask correct?
        # Yes! Use the mask associated with key/value, not query. (query, key, value)
        # Additionally, there is no need to do subsequent masking, since each word has the right to see
        # all the video info.
        dec_enc_attention_output = self.dec_enc_attention(
            attention_output, enc_outputs, enc_outputs, enc_mask.unsqueeze(1))  # (N, Lt, D)
        dec_enc_attention_output = self.norm2(attention_output + dec_enc_attention_output)  # (N, Lt, D)

        # 3, linear + add_norm
        dec_enc_attention_output = self.output(dec_enc_attention_output, dec_enc_attention_output)  # (N, Lt, D)
        return dec_enc_attention_output  # (N, Lt, D)


class BertDecoderNoMemoryUntied(nn.Module):
    def __init__(self, config):
        super(BertDecoderNoMemoryUntied, self).__init__()
        self.layer = nn.ModuleList([BertDecoderLayerNoMemoryUntied(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, dec_hidden_states, dec_mask, enc_outputs, enc_mask,
                diagonal_mask=True, output_all_encoded_layers=False):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            dec_hidden_states = layer_module(
                dec_hidden_states, dec_mask, enc_outputs, enc_mask, diagonal_mask=True)
            if output_all_encoded_layers:
                all_encoder_layers.append(dec_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(dec_hidden_states)
        return all_encoder_layers


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None, \
                "bert_model_embedding_weights should not be None " \
                "when setting --share_wd_cls_weight flag to be true"
            assert config.hidden_size == bert_model_embedding_weights.size(1), \
                "hidden size has be the same as word embedding size when " \
                "sharing word embedding weight and classifier weight"
            self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                     bert_model_embedding_weights.size(0),
                                     bias=False)
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)


class EntitiyReasoningNetwork(nn.Module):
    def __init__(self, config):
        super(EntitiyReasoningNetwork, self).__init__()
        self.config = config

        self.lstm_hidden_size = config.lstm_hidden_size
        self.entity_hidden_size = self.lstm_hidden_size
        self.action_hidden_size = self.lstm_hidden_size
        self.action_emb_size = config.word_vec_size

        #self.step_wise_encoder = nn.LSTM(config.hidden_size, config.lstm_hidden_size, bidirectional=True, batch_first=True)

        # action selector
        self.action_selector = nn.Sequential(nn.Linear(self.action_hidden_size, self.action_hidden_size),
                                             nn.ReLU(),
                                             nn.Dropout(p=0.4),
                                             nn.Linear(self.action_hidden_size, config.action_vocab_size),
                                             nn.Sigmoid())
        self.action_embeddings = nn.Embedding(config.action_vocab_size, config.word_vec_size)


        # entity selector
        self.W1 = nn.Sequential(nn.Linear(self.entity_hidden_size, self.entity_hidden_size),
                                          nn.ReLU())
        self.W2 = nn.Linear(self.entity_hidden_size + config.action_vocab_size, self.entity_hidden_size)
        self.W3 = nn.Linear(self.entity_hidden_size, 3)

        # updater
        self.W4 = nn.Linear(self.config.word_vec_size, 1)

        
    def set_pretrained_embedding(self, pretrained_embedding, freeze):
        self.action_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze)
        self.action_embeddings = self.action_embeddings.weight
    
    def forward(self, video_vectors, entity_vectors):
        #video_vectors = self.step_wise_encoder(video_vectors, torch.ones(1, video_vectors.shape[1]).cuda())[-1].squeeze(0)
        video_vectors = video_vectors.squeeze(0)
        step_num = video_vectors.shape[0]
        entity_num = entity_vectors.shape[0]

        ac_probs = torch.zeros((step_num, self.config.action_vocab_size)).cuda()
        ent_probs = torch.zeros((step_num, entity_num)).cuda()
        
        step_action_vectors = torch.zeros((step_num, self.action_emb_size)).cuda()
        step_entity_vectors = torch.zeros((step_num, self.entity_hidden_size)).cuda()
        step_all_entity_vectors = torch.zeros((step_num, entity_num, self.entity_hidden_size)).cuda()

        prev_ent_attn = torch.zeros(entity_num).cuda()

        for step_idx in range(step_num):
            #v_ac, v_ent = video_vectors[step_idx][:self.action_hidden_size], video_vectors[step_idx][self.action_hidden_size:]
            v_ac = video_vectors[step_idx]

            # (1) action selector
            ac_prob = self.action_selector(v_ac)
            bar_ft = (ac_prob / ac_prob.sum()).unsqueeze(0) @ self.action_embeddings # Eq. (1)

            # (2) entity selector
            # sentence attention
            hat_ht = self.W1(v_ac) # Eq. (2)
            ent_prob = F.sigmoid(entity_vectors @ self.W2(torch.cat([hat_ht, ac_prob]))) # Eq. (2)
            choice_dist = F.softmax(self.W3(hat_ht), dim=-1) # Eq. (3)
            attn_dist = choice_dist[0] * ent_prob + choice_dist[1] * prev_ent_attn + choice_dist[2] * torch.zeros(entity_num).cuda() # Eq. (3)

            # recurrent attention
            bar_et = (attn_dist / attn_dist.sum()).unsqueeze(0) @ entity_vectors # Eq. (4), (5)

            # (3) update entities
            kt = F.relu(self.W4(bar_ft) @ bar_et) # Eq. (6)
            entity_vectors = attn_dist.unsqueeze(0).T @ kt + (1 - attn_dist).unsqueeze(0).T * entity_vectors # Eq. (7), update
            prev_ent_attn = ent_prob

            # store them
            ac_probs[step_idx] = ac_prob
            ent_probs[step_idx] = ent_prob
            step_action_vectors[step_idx] = bar_ft
            step_entity_vectors[step_idx] = bar_et # bar_etでもいいような...
            step_all_entity_vectors[step_idx] = entity_vectors

        # compute loss
        return ent_probs, ac_probs, step_entity_vectors, step_all_entity_vectors, step_action_vectors


class StateAwareRecursiveTransformer(nn.Module):
    def __init__(self, config):
        super(StateAwareRecursiveTransformer, self).__init__()
        self.config = config

        self.ingredient_embeddings = BertEmbeddingsIngredientsUntied(config)
        self.video_embeddings = BertEmbeddingsVideoUntied(config)
        self.text_embeddings = BertEmbeddingsTextUntied(config)
        self.token_type_embeddings = nn.Embedding(4, config.hidden_size) # 4 --> 0 (PAD), 1 (CLS), 2 (SEP), 3 (VID)
        
        # encoder
        self.encoder = BertEncoderNoMemoryUntied(config)
        self.step_wise_encoder = BertEncoderNoMemoryUntied(config)
        self.step_positional_encoding = PositionEncoding(n_filters=config.hidden_size, max_len=50)

        # visual simulator
        self.reasoner = EntitiyReasoningNetwork(config)

        # decoder
        self.Wac = nn.Sequential(
                nn.Linear(config.word_vec_size, config.hidden_size),
                nn.ReLU()
                )
        self.Went = nn.Sequential(
                nn.Linear(config.lstm_hidden_size, config.hidden_size),
                nn.ReLU()
                )
        self.decoder = BertDecoderNoMemoryUntied(config)
        decoder_classifier_weight = self.embeddings.word_embeddings.weight \
            if self.config.share_wd_cls_weight else None
        self.decoder_classifier = BertLMPredictionHead(config, decoder_classifier_weight)

        # pointer generator network
        self.Wing = nn.Linear(config.intermediate_size, config.intermediate_size)
        self.pgen_linear = nn.Sequential(nn.Linear(config.intermediate_size*2, 1),
                                         nn.Sigmoid())

        # textual re-simulator
        self.eps = 1e-12
        self.recipe_encoder = nn.LSTM(config.word_vec_size, config.hidden_size, batch_first=True, bidirectional=True)
        self.recipe_reasoner = EntitiyReasoningNetwork(config)

        # loss functions
        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion = nn.BCELoss(reduction="sum")
        self.ac_criterion = AsymmetricLoss()
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_step(self, input_ids, video_features, input_masks):
        """single step forward in the recursive structure"""
        video_embeddings = self.video_embeddings(video_features[:, :self.config.max_v_len])
        token_type_embeddings = self.token_type_embeddings(input_ids[:, :self.config.max_v_len])
        video_embeddings = video_embeddings + token_type_embeddings
        video_masks = input_masks[:, :self.config.max_v_len]
        encoder_outputs = self.encoder(video_embeddings, video_masks, diagonal_mask=False)[-1]
        return encoder_outputs

    def pointer_generator_network(self, decoder_outputs, ingr_vectors, ingr_dict, extra_zero):
        #ingr_vectors = ingr_vectors.unsqueeze(0).repeat(decoder_outputs.shape[0], 1, 1).transpose(1, 0).contiguous()
        ingr_vectors = ingr_vectors.transpose(1, 0).contiguous()
        attn_dist = torch.einsum('abc,bdc->abd', self.Wing(ingr_vectors), decoder_outputs)
        attn_dist = F.softmax(attn_dist, dim=0)
        t_th_attended_ingr_vectors = torch.einsum("abc,abd -> bcd", attn_dist, ingr_vectors)

        p_gen = torch.cat([decoder_outputs, t_th_attended_ingr_vectors], dim=2)
        p_gen = self.pgen_linear(p_gen)

        vocab_dist = self.decoder_classifier(decoder_outputs)
        vocab_dist = F.softmax(vocab_dist, dim=-1)
        vocab_dist = p_gen * vocab_dist
        attn_dist =  attn_dist * (1 - p_gen).squeeze(2)

        if extra_zero == 0:
            for ingr_num, ingr_ids in ingr_dict.items():
                for ingr_id in ingr_ids:
                    vocab_dist[:, :, ingr_id] += attn_dist[ingr_num] / len(ingr_ids)
            return vocab_dist
        else:
            seq_num, word_num = vocab_dist.shape[0], vocab_dist.shape[1]
            extra_zero_dist = torch.zeros((seq_num, word_num, extra_zero)).cuda()
            vocab_dist = torch.cat([vocab_dist, extra_zero_dist], dim=-1)
            for ingr_num, ingr_ids in ingr_dict.items():
                for ingr_id in ingr_ids:
                    vocab_dist[:, :, ingr_id] += attn_dist[ingr_num] / len(ingr_ids)
            return vocab_dist

    def decode(self, text_input_ids, text_masks, 
               text_input_labels, ga_ingr_vectors, ga_step_vectors,
               entity_vectors, action_vectors, step_entity_vectors,
               ith_ingr_dict, ith_extra_zero):
        """
        Args:
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits
            text_input_labels: (N, Lt)  with `-1` on ignored positions
            encoder_outputs: (N, Lv, D)
            encoder_masks: (N, Lv)
        """

        if self.config.model_mode == "full": # VIVT
            entity_vectors = self.Went(entity_vectors)
            action_vectors = self.Wac(action_vectors)

            ga_step_vectors = ga_step_vectors.transpose(1, 0).contiguous()
            entity_vectors = entity_vectors.unsqueeze(1)
            action_vectors = action_vectors.unsqueeze(1)

            ga_inputs = torch.cat([ga_step_vectors, entity_vectors, action_vectors], dim=1)
            ga_mask = torch.ones((ga_inputs.shape[0], ga_inputs.shape[1])).cuda()

            text_embeddings = self.text_embeddings(text_input_ids)

            decoder_outputs = self.decoder(text_embeddings, text_masks, ga_inputs, ga_mask, diagonal_mask=True)[-1]
            
            # pointer-generator
            prediction_scores = self.pointer_generator_network(decoder_outputs,
                                                               step_entity_vectors,
                                                               ith_ingr_dict,
                                                               ith_extra_zero)

            extended_vocab_size = prediction_scores.shape[-1]
            caption_loss = self.loss_func(prediction_scores.view(-1, extended_vocab_size), text_input_labels.view(-1))
            return caption_loss, prediction_scores, decoder_outputs
        
        elif self.config.model_mode == "reason_copy": # VIV
            entity_vectors = self.Went(entity_vectors)
            action_vectors = self.Wac(action_vectors)

            ga_step_vectors = ga_step_vectors.transpose(1, 0).contiguous()
            entity_vectors = entity_vectors.unsqueeze(1)
            action_vectors = action_vectors.unsqueeze(1)

            ga_inputs = torch.cat([ga_step_vectors, entity_vectors, action_vectors], dim=1)
            ga_mask = torch.ones((ga_inputs.shape[0], ga_inputs.shape[1])).cuda()

            text_embeddings = self.text_embeddings(text_input_ids)
            decoder_outputs = self.decoder(text_embeddings, text_masks, ga_inputs, ga_mask, diagonal_mask=True)[-1]

            prediction_scores = self.pointer_generator_network(decoder_outputs,
                                                               step_entity_vectors,
                                                               ith_ingr_dict,
                                                               ith_extra_zero)
            extended_vocab_size = prediction_scores.shape[-1]
            caption_loss = self.loss_func(prediction_scores.view(-1, extended_vocab_size), text_input_labels.view(-1))
            return caption_loss, prediction_scores, decoder_outputs
        
        elif self.config.model_mode == "copy": # VI
            ga_step_vectors = ga_step_vectors.transpose(1, 0).contiguous()
            step_num = ga_step_vectors.shape[0]
            ga_ingr_verbose_vectors = ga_ingr_vectors.clone().unsqueeze(0).repeat(step_num, 1, 1)
            ga_ingr_vectors = ga_ingr_vectors.mean(dim=0).unsqueeze(0).unsqueeze(0).repeat(step_num, 1, 1)
            ga_inputs = torch.cat([ga_step_vectors, ga_ingr_vectors], dim=1)
            ga_mask = torch.ones([ga_inputs.shape[0], ga_inputs.shape[1]]).cuda()

            text_embeddings = self.text_embeddings(text_input_ids)
            decoder_outputs = self.decoder(text_embeddings, text_masks, ga_inputs, ga_mask, diagonal_mask=True)[-1]

            prediction_scores = self.pointer_generator_network(decoder_outputs,
                                                               ga_ingr_verbose_vectors,
                                                               ith_ingr_dict,
                                                               ith_extra_zero)

            extended_vocab_size = prediction_scores.shape[-1]
            caption_loss = self.loss_func(prediction_scores.view(-1, extended_vocab_size), text_input_labels.view(-1))
            return caption_loss, prediction_scores, decoder_outputs

        else: # V
            ga_inputs = ga_step_vectors.transpose(1, 0).contiguous()
            ga_mask = torch.ones([ga_inputs.shape[0], ga_inputs.shape[1]]).cuda()

            text_embeddings = self.text_embeddings(text_input_ids)
            decoder_outputs = self.decoder(text_embeddings, text_masks, ga_inputs, ga_mask, diagonal_mask=True)[-1]
            prediction_scores = self.decoder_classifier(decoder_outputs)
            prediction_scores = F.softmax(prediction_scores, dim=-1)
            text_input_labels[text_input_labels >= self.config.vocab_size] = self.config.unk_id
            caption_loss = self.loss_func(prediction_scores.view(-1, prediction_scores.shape[-1]), text_input_labels.view(-1))
            return caption_loss, prediction_scores, None

    def reconstruct(self, prediction_scores, text_mask, ga_ingr_vectors):
        bow_embeddings = F.gumbel_softmax(torch.log(prediction_scores + self.eps), tau=self.config.temperature, hard=True)[:, :, :self.config.vocab_size] @ self.text_embeddings.word_embeddings.weight
        rm_decoder_outputs = torch.zeros((bow_embeddings.shape[0], bow_embeddings.shape[-1])).cuda()
        for t_idx, t_mask in enumerate(text_mask):
            rm_decoder_outputs[t_idx] = bow_embeddings[t_idx][t_mask == 1].mean(dim=0)
        self.recipe_encoder.flatten_parameters()
        rm_decoder_outputs = self.recipe_encoder(rm_decoder_outputs.unsqueeze(0))[0]
        rm_decoder_outputs = rm_decoder_outputs[:, :, :self.config.hidden_size] + rm_decoder_outputs[:, :, self.config.hidden_size:]
        return self.recipe_reasoner(rm_decoder_outputs, ga_ingr_vectors)
    
    def forward(self, input_ids_list, video_features_list, input_masks_list,
                token_type_ids_list, input_labels_list, ingr_input_ids,
                ingr_masks, ingr_sep_masks, batch_step_num, ingr_id_dict, extra_zeros,
                alignments, actions, return_memory=False, predict=False):

        # initialize state of entities
        ingredient_outputs = self.ingredient_embeddings(ingr_input_ids, ingr_sep_masks)

        # 2. encode video-text unified vectors (step-level)
        step_size = len(input_ids_list)
        encoded_outputs_list = []
        for idx in range(step_size):
            encoded_layer_outputs = self.forward_step(input_ids_list[idx], 
                                                      video_features_list[idx],
                                                      input_masks_list[idx])
            encoded_outputs_list.append(encoded_layer_outputs)
        
        prediction_scores_list = []
        entity_prob_list = []
        action_prob_list = []
        caption_loss = 0.

        # only for predict=True
        memory_dict_list = []

        entity_loss = 0.
        action_loss = 0.

        # textual re-simulator
        re_entity_loss = 0.
        re_action_loss = 0.

        for batch_idx in range(len(batch_step_num)):
            # encode
            step_num = batch_step_num[batch_idx]
            ith_encoded_outputs_list = [x[batch_idx] for x in encoded_outputs_list[:step_num]]
            ga_step_vectors = torch.stack(ith_encoded_outputs_list)
            ga_step_vectors = self.step_positional_encoding(ga_step_vectors[:, 0, :].unsqueeze(0)) # use [CLS]
            ga_step_vectors = self.step_wise_encoder(ga_step_vectors, torch.ones(1, ga_step_vectors.shape[1]).cuda())[-1]

            ingr_num = (ingr_sep_masks[batch_idx] == 1).sum().item()
            ga_ingr_vectors = ingredient_outputs[batch_idx, :ingr_num]

            ith_input_ids_list = [x[batch_idx][self.config.max_v_len:] for x in input_ids_list[:step_num]]
            ith_input_masks_list = [x[batch_idx][self.config.max_v_len:] for x in input_masks_list[:step_num]]
            ith_input_labels_list = [x[batch_idx][self.config.max_v_len:] for x in input_labels_list[:step_num]]

            ith_input_ids_list = torch.stack(ith_input_ids_list)
            ith_input_masks_list = torch.stack(ith_input_masks_list)
            ith_input_labels_list = torch.stack(ith_input_labels_list)

            # pointer generator
            ith_ingr_dict = ingr_id_dict[batch_idx]
            ith_extra_zero = extra_zeros[batch_idx]

            if self.config.model_mode == "full": # VIVT
                # reasoning networks
                entity_probs, action_probs, entity_vectors, step_entity_vectors, action_vectors = self.reasoner(ga_step_vectors, ga_ingr_vectors)

                tmp_caption_loss, prediction_scores, decoder_outputs = self.decode(ith_input_ids_list,
                                                                                  ith_input_masks_list,
                                                                                  ith_input_labels_list,
                                                                                  ga_ingr_vectors,
                                                                                  ga_step_vectors,
                                                                                  entity_vectors,
                                                                                  action_vectors,
                                                                                  step_entity_vectors,
                                                                                  ith_ingr_dict,
                                                                                  ith_extra_zero)

                # reprediction
                re_entity_probs, re_action_probs, re_entity_vectors, \
                    re_step_entity_vectors, re_action_vectors = self.reconstruct(prediction_scores, ith_input_masks_list, ga_ingr_vectors)

                # Notice: calculate loss if actions are detected
                tmp_entity_loss = self.criterion(entity_probs, alignments[batch_idx])
                action_bool = (actions[batch_idx] == 1).any(dim=1)
                tmp_action_loss = self.ac_criterion(action_probs[action_bool], actions[batch_idx][action_bool])

                # reprediction
                re_tmp_entity_loss = self.criterion(re_entity_probs, alignments[batch_idx])
                re_tmp_action_loss = self.ac_criterion(re_action_probs[action_bool], actions[batch_idx][action_bool])

                caption_loss += tmp_caption_loss
                entity_loss += tmp_entity_loss
                action_loss += tmp_action_loss

                re_entity_loss += re_tmp_entity_loss
                re_action_loss += re_tmp_action_loss

                prediction_scores_list.append(prediction_scores)
                entity_prob_list.append(entity_probs)
                action_prob_list.append(action_probs)
                memory_dict_list.append({
                    "entity_probs" : entity_probs,
                    "action_probs" : action_probs,
                    "entity_vectors" : [ga_ingr_vectors, step_entity_vectors],
                    "re_pred_entity_probs" : re_entity_probs,
                    "re_pred_action_probs" : re_action_probs,
                    "re_entity_vectors" : re_step_entity_vectors
                    })

            elif self.config.model_mode == "reason_copy": # VIV
                # reasoning networks
                entity_probs, action_probs, entity_vectors, step_entity_vectors, action_vectors = self.reasoner(ga_step_vectors, ga_ingr_vectors)

                tmp_caption_loss, prediction_scores, decoder_outputs = self.decode(ith_input_ids_list,
                                                                                  ith_input_masks_list,
                                                                                  ith_input_labels_list,
                                                                                  ga_ingr_vectors,
                                                                                  ga_step_vectors,
                                                                                  entity_vectors,
                                                                                  action_vectors,
                                                                                  step_entity_vectors,
                                                                                  ith_ingr_dict,
                                                                                  ith_extra_zero)
                
                
                tmp_entity_loss = self.criterion(entity_probs, alignments[batch_idx])
                action_bool = (actions[batch_idx] == 1).any(dim=1)
                tmp_action_loss = self.ac_criterion(action_probs[action_bool], actions[batch_idx][action_bool])

                caption_loss += tmp_caption_loss
                entity_loss += tmp_entity_loss
                action_loss += tmp_action_loss

                prediction_scores_list.append(prediction_scores)
                entity_prob_list.append(entity_probs)
                action_prob_list.append(action_probs)
                memory_dict_list.append({
                    "entity_probs" : entity_probs,
                    "action_probs" : action_probs,
                    "entity_vectors" : [ga_ingr_vectors, step_entity_vectors],
                    })
            
            elif self.config.model_mode == "copy": # VI
                tmp_caption_loss, prediction_scores, _ = self.decode(ith_input_ids_list,
                                                                     ith_input_masks_list,
                                                                     ith_input_labels_list,
                                                                     ga_ingr_vectors,
                                                                     ga_step_vectors,
                                                                     None, None, None,
                                                                     ith_ingr_dict,
                                                                     ith_extra_zero)
                caption_loss += tmp_caption_loss
                prediction_scores_list.append(prediction_scores)

            else: # V
                tmp_caption_loss, prediction_scores, _ = self.decode(ith_input_ids_list,
                                                                     ith_input_masks_list,
                                                                     ith_input_labels_list,
                                                                     None,
                                                                     ga_step_vectors,
                                                                     None, None, None, None, None)
                caption_loss += tmp_caption_loss
                prediction_scores_list.append(prediction_scores)


        if predict:
            return memory_dict_list, entity_prob_list, action_prob_list
        else:
            total_loss = caption_loss + entity_loss + action_loss + self.config.lambda_ * (re_entity_loss + re_action_loss)
            return total_loss, prediction_scores_list, entity_prob_list, action_prob_list

# remind me of what the configs are
base_config = edict(
    hidden_size=768,
    vocab_size=None,  # get from word2idx
    video_feature_size=2048,
    max_position_embeddings=None,  # get from max_seq_len
    max_v_len=100,  # max length of the videos
    max_t_len=30,  # max length of the text
    n_memory_cells=10,  # memory size will be (n_memory_cells, D)
    type_vocab_size=2,
    layer_norm_eps=1e-12,  # bert layernorm
    hidden_dropout_prob=0.1,  # applies everywhere except attention
    num_hidden_layers=2,  # number of transformer layers
    attention_probs_dropout_prob=0.1,  # applies only to self attention
    intermediate_size=768,  # after each self attention
    num_attention_heads=12,
    memory_dropout_prob=0.1
)
