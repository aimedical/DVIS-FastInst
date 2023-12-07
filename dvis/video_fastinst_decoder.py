import einops
import torch
from detectron2.config import configurable
from fastinst.modeling.transformer_decoder import FastInstDecoder
from torch import nn
from torch.nn import functional as F

from fastinst.modeling.transformer_decoder.utils import TRANSFORMER_DECODER_REGISTRY


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoFastInstDecoder_dvis(FastInstDecoder):

    @configurable
    def __init__(
            self,
            in_channels,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            num_aux_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            # video related
            num_frames: int,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_aux_queries=num_aux_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
        )
        self.num_frames = num_frames

    @classmethod
    def from_config(cls, cfg, in_channels, input_shape):
        ret = {}
        ret["in_channels"] = in_channels

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.FASTINST.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.FASTINST.NUM_OBJECT_QUERIES
        ret["num_aux_queries"] = cfg.MODEL.FASTINST.NUM_AUX_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.FASTINST.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.FASTINST.DIM_FEEDFORWARD

        ret["dec_layers"] = cfg.MODEL.FASTINST.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.FASTINST.PRE_NORM

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM

        return ret

    # referenced https://github.com/junjiehe96/FastInst/blob/main/fastinst/modeling/transformer_decoder/fastinst_decoder.py
    # referenced https://github.com/zhang-tao-whu/DVIS/blob/main/dvis/video_mask2former_transformer_decoder.py
    def forward(self, x, _, targets=None):
        bs = x[0].shape[0]
        proposal_size = x[1].shape[-2:]
        pixel_feature_size = x[2].shape[-2:]

        pixel_pos_embeds = F.interpolate(self.meta_pos_embed, size=pixel_feature_size,
                                         mode="bilinear", align_corners=False)
        proposal_pos_embeds = F.interpolate(self.meta_pos_embed, size=proposal_size,
                                            mode="bilinear", align_corners=False)

        pixel_features = x[2].flatten(2).permute(2, 0, 1)
        pixel_pos_embeds = pixel_pos_embeds.flatten(2).permute(2, 0, 1)

        query_features, query_pos_embeds, query_locations, proposal_cls_logits = self.query_proposal(
            x[1], proposal_pos_embeds
        )
        query_features = query_features.permute(2, 0, 1)
        query_pos_embeds = query_pos_embeds.permute(2, 0, 1)
        if self.num_aux_queries > 0:
            aux_query_features = self.empty_query_features.weight.unsqueeze(1).repeat(1, bs, 1)
            aux_query_pos_embed = self.empty_query_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            query_features = torch.cat([query_features, aux_query_features], dim=0)
            query_pos_embeds = torch.cat([query_pos_embeds, aux_query_pos_embed], dim=0)

        outputs_class, outputs_mask, attn_mask, _, _ = self.forward_prediction_heads(
            query_features, pixel_features, pixel_feature_size, -1, return_attn_mask=True
        )
        predictions_class = [outputs_class]
        predictions_mask = [outputs_mask]
        predictions_matching_index = [None]
        query_feature_memory = [query_features]
        pixel_feature_memory = [pixel_features]

        for i in range(self.num_layers):
            query_features, pixel_features = self.forward_one_layer(
                query_features, pixel_features, query_pos_embeds, pixel_pos_embeds, attn_mask, i
            )
            if i < self.num_layers - 1:
                outputs_class, outputs_mask, attn_mask, _, _ = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, i, return_attn_mask=True,
                )
            else:
                outputs_class, outputs_mask, _, matching_indices, gt_attn_mask = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, i,
                    return_gt_attn_mask=self.training, targets=targets, query_locations=query_locations
                )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_matching_index.append(None)
            query_feature_memory.append(query_features)
            pixel_feature_memory.append(pixel_features)

        guided_predictions_class = []
        guided_predictions_mask = []
        guided_predictions_matching_index = []
        if self.training:
            for i in range(self.num_layers):
                query_features, pixel_features = self.forward_one_layer(
                    query_feature_memory[i + 1], pixel_feature_memory[i + 1], query_pos_embeds,
                    pixel_pos_embeds, gt_attn_mask, i
                )

                outputs_class, outputs_mask, _, _, _ = self.forward_prediction_heads(
                    query_features, pixel_features, pixel_feature_size, idx_layer=i
                )

                guided_predictions_class.append(outputs_class)
                guided_predictions_mask.append(outputs_mask)
                guided_predictions_matching_index.append(matching_indices)

        predictions_class = guided_predictions_class + predictions_class
        predictions_mask = guided_predictions_mask + predictions_mask
        predictions_matching_index = guided_predictions_matching_index + predictions_matching_index

        # if training DVIS, expand BT to B, T
        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)

        if self.training:
            for i in range(len(predictions_class)):
                predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) q c -> b t q c', t=t)
        else: # if DVIS
            predictions_class[-1] = einops.rearrange(predictions_class[-1], '(b t) q c -> b t q c', t=t)

        pred_embds = self.decoder_query_norm_layers[-1](query_features[:self.num_queries])
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)

        mask_features = pixel_features # (hw, bt, c)

        out = {
            'proposal_cls_logits': proposal_cls_logits,
            'query_locations': query_locations,
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_matching_indices': predictions_matching_index[-1], # 常にNoneになる謎の返り値
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask, predictions_matching_index, query_locations
            ),
            'pred_embds': pred_embds,
            'mask_features': mask_features,
            'pixel_feature_size': pixel_feature_size,
        }
        return out