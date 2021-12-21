# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Updated by Guangsheng Bao on 11/11/2020


from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, add_global_attn=False):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.need_attn_entropy = getattr(args, 'doc_attn_entropy', False)

        # Guangsheng Bao:
        # XXX_attn_local is for local sentence (Group Attention), while XXX_attn_global is for global (Global Attention)
        self.self_attn_local = self.build_self_attention(self.embed_dim, args)
        if add_global_attn:
            self.self_attn_global = self.build_self_attention(self.embed_dim, args)
            self.self_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim, args.encoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

        # Guangsheng Bao: rename attention since name changed for doc.
        attn_name_map = {"self_attn": "self_attn_local"}
        for old, new in attn_name_map.items():
            for m in ["in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"]:
                k = "{}.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

        # Guangsheng Bao: initialize global attention using local attention
        attn_name_map = {}
        if getattr(self, 'self_attn_global', None) is not None:
            attn_name_map["self_attn_local"] = "self_attn_global"

        key_global = "doc_attn_global_from_attn_local"
        if key_global in state_dict and state_dict[key_global]:
            for old, new in attn_name_map.items():
                for m in ["in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias",
                          "k_proj.weight", "k_proj.bias", "q_proj.weight", "q_proj.bias", "v_proj.weight", "v_proj.bias"]:
                    k_old = "{}.{}.{}".format(name, old, m)
                    k_new = "{}.{}.{}".format(name, new, m)
                    if k_old in state_dict and k_new not in state_dict:
                        state_dict[k_new] = state_dict[k_old].clone()

    def forward(self, x, padding_mask, local_attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            local_attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `local_attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original local_attn_mask = 1, becomes -1e8
        # anything in original local_attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # if local_attn_mask is not None:
        #     local_attn_mask = local_attn_mask.masked_fill(local_attn_mask.to(torch.bool), -1e8)

        attn = {}
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x_local, attn_local = self.self_attn_local(
            query=x,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            attn_mask=local_attn_mask,
            need_weights=self.need_attn_entropy,
        )
        if attn_local is not None:
            attn['encoder_self_local'] = utils.attn_entropy(attn_local, mean_dim=1)

        # for partial mode
        if getattr(self, 'self_attn_global', None) is not None:
            x_global, attn_global = self.self_attn_global(
                query=x,
                key=x,
                value=x,
                key_padding_mask=padding_mask,
                need_weights=self.need_attn_entropy,
            )
            if attn_global is not None:
                attn['encoder_self_global'] = utils.attn_entropy(attn_global, mean_dim=1)
            # merge with local
            g = self.self_attn_gate(torch.cat([x_local, x_global], dim=-1))
            x = x_local * g + x_global * (1 - g)
        else:
            x = x_local

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False,
        dec_add_global_attn=False, crs_add_global_attn=False,
        add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.dropout
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.need_attn_entropy = getattr(args, 'doc_attn_entropy', False)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        # Guangsheng Bao:
        # XXX_attn_local is for local sentence (Group Attention), while XXX_attn_global is for global (Global Attention)
        assert no_encoder_attn == False
        self.self_attn_local = self.build_self_attention(self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        self.encoder_attn_local = self.build_encoder_attention(self.embed_dim, args.decoder_attention_heads, args)
        if dec_add_global_attn:
            self.self_attn_global = self.build_self_attention(self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
            self.self_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())
        if crs_add_global_attn:
            self.encoder_attn_global = self.build_encoder_attention(self.embed_dim, args.decoder_attention_heads, args)
            self.encoder_attn_gate = nn.Sequential(Linear(self.embed_dim * 2, self.embed_dim), nn.Sigmoid())

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)

        self.fc1 = self.build_fc1(
            self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.normalize_before = args.decoder_normalize_before

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, attn_heads, args):
        return MultiheadAttention(
            embed_dim,
            attn_heads,
            qdim=getattr(args, "encoder_embed_dim", None),
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            odim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def upgrade_state_dict_named(self, state_dict, name):
        # Guangsheng Bao: rename attention since name changed for doc.
        attn_name_map = {"self_attn": "self_attn_local", "encoder_attn": "encoder_attn_local"}
        for old, new in attn_name_map.items():
            for m in ["in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"]:
                k = "{}.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

        # Guangsheng Bao: initialize global attention using local attention
        attn_name_map = {}
        if getattr(self, 'self_attn_global', None) is not None:
            attn_name_map["self_attn_local"] = "self_attn_global"
        if getattr(self, 'encoder_attn_global', None) is not None:
            attn_name_map["encoder_attn_local"] = "encoder_attn_global"

        key_global = "doc_attn_global_from_attn_local"
        if key_global in state_dict and state_dict[key_global] and len(attn_name_map) > 0:
            for old, new in attn_name_map.items():
                for m in ["in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias",
                          "k_proj.weight", "k_proj.bias", "q_proj.weight", "q_proj.bias", "v_proj.weight", "v_proj.bias"]:
                    k_old = "{}.{}.{}".format(name, old, m)
                    k_new = "{}.{}.{}".format(name, new, m)
                    if k_old in state_dict and k_new not in state_dict:
                        state_dict[k_new] = state_dict[k_old].clone()

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_local_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        local_attn_mask: Optional[torch.Tensor] = None,
        global_attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        attn = {}
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x_local, attn_local = self.self_attn_local(
            query=x, key=x, value=x,
            key_padding_mask=padding_mask,
            incremental_state=incremental_state,
            attn_mask=local_attn_mask,
            need_weights=self.need_attn_entropy,
        )
        if attn_local is not None:
            attn['decoder_self_local'] = utils.attn_entropy(attn_local, mean_dim=1)

        # for partial mode, we combine local and global attention
        if getattr(self, 'self_attn_global', None) is not None:
            x_global, attn_global = self.self_attn_global(
                query=x, key=x, value=x,
                key_padding_mask=padding_mask,
                incremental_state=incremental_state,
                attn_mask=global_attn_mask,
                need_weights=self.need_attn_entropy,
            )
            if attn_global is not None:
                attn['decoder_self_global'] = utils.attn_entropy(attn_global, mean_dim=1)
            # merge with local
            g = self.self_attn_gate(torch.cat([x_local, x_global], dim=-1))
            x = x_local * g + x_global * (1 - g)
        else:
            x = x_local

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn_local is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x_local, attn_local = self.encoder_attn_local(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                attn_mask=encoder_local_mask,
                need_weights=self.need_attn_entropy,
            )
            if attn_local is not None:
                attn['decoder_cross_local'] = utils.attn_entropy(attn_local, mean_dim=1)

            # for partial mode, we combine local and global attention
            if getattr(self, 'encoder_attn_global', None) is not None:
                x_global, attn_global = self.encoder_attn_global(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=self.need_attn_entropy,
                )
                if attn_global is not None:
                    attn['decoder_cross_global'] = utils.attn_entropy(attn_global, mean_dim=1)
                # merge with local
                g = self.encoder_attn_gate(torch.cat([x_local, x_global], dim=-1))
                x = x_local * g + x_global * (1 - g)
            else:
                x = x_local

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            assert False
            saved_state_local = self.self_attn_local._get_input_buffer(incremental_state)
            assert saved_state_local is not None
            if padding_mask is not None:
                self_attn_state = [
                    saved_state_local["prev_key"],
                    saved_state_local["prev_value"],
                    saved_state_local["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state_local["prev_key"], saved_state_local["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
