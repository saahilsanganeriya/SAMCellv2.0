import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List, Tuple
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam2_utils import MLP, NO_OBJ_SCORE

from transformers.models.sam

class MaskDecoder3Ch(SamMaskDecoder):
    """
    Subclass of the SAM 2.1 `MaskDecoder` that produces exactly 3 output channels
    from a single mask token, instead of producing multiple single-channel masks.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: nn.Module = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        # we skip some arguments from original for brevity
        **kwargs,
    ):
        """
        Minimal subset of the original init. We set `num_mask_tokens=1`.
        We also create exactly 1 MLP in `self.output_hypernetworks_mlps`, but it
        produces `3*(transformer_dim//8)` so we can reshape to 3 channels.
        """
        super().__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=0,  # not used
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
            use_high_res_features=use_high_res_features,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            **kwargs,
        )

        # 1) override the number of mask tokens to 1
        self.num_mask_tokens = 1  # only 1 token

        # 2) Rebuild the embeddings. We'll keep a single token for iou_token,
        #    and a single mask token => total = 2 if we keep iou_token logic
        #    (the original code: iou_token + (num_multimask_outputs+1) mask tokens).
        #    We'll do iou_token = 1, mask_tokens = 1 => total 2 tokens in embedding.
        # If you do *not* want IoU predictions, you can skip iou_token logic entirely.
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # For the final MLP, we produce 3 * (transformer_dim//8) so we can reshape
        # it into (b, 1, 3, transformer_dim//8) at runtime.
        # So we must set out_features = (transformer_dim // 8) * 3, i.e. 3 times bigger
        # than usual. The original code does MLP(..., out_features=transformer_dim//8, ...)
        # for each mask token. We'll do 3x that to get 3 channels from the single token.
        mlp_out = (transformer_dim // 8) * 3
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, mlp_out, 3)  # 3-layer MLP
        ])

        # For iou_prediction_head, we still produce 1 channel => shape [B,1]
        # or you can skip it if you do not need IoU.
        # The parent class made `self.iou_prediction_head` automatically, so we'll keep it.

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Very similar to the parent's `forward`, but we've only got 1 mask token => 3-ch.
        """
        # We'll basically copy the parent's code but reduce the `num_mask_tokens` logic.
        # Step 1: build the output tokens
        #   If we keep iou_token, then first embedding => iou_token.weight
        #   Then the single mask token => mask_tokens.weight
        bsz = sparse_prompt_embeddings.size(0)  # batch size

        # If self.pred_obj_scores is True, the original code also has object_score_token logic,
        # we skip it for brevity. We'll just do iou_token + mask_token:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # shape => [1+1, transformer_dim] => [2, C]
        output_tokens = output_tokens.unsqueeze(0).expand(bsz, -1, -1)
        # shape => [B, 2, C]

        # Then cat the sparse prompts (e.g. points) after that
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Step 2: we might need to replicate image_embeddings if repeat_image is True
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings

        # shape => [B, C, H, W]
        b, c, h, w = src.shape
        # Step 3: run the transformer's forward
        hs, src = self.transformer(src, image_pe, tokens)
        # parent logic => iou_token_out = hs[:, 0, :], mask_tokens_out = hs[:, 1 : 1 + self.num_mask_tokens, :]
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : 2, :]  # since self.num_mask_tokens=1

        # Step 4: upscaling
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            # If parent uses high-res features
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # shape => [B, C', Hup, Wup], typically (C' = transformer_dim // 8)
        # Step 5: apply the per-token hypernet MLP
        # We only have 1 token => index=0
        mlp = self.output_hypernetworks_mlps[0]
        hyper_in = mlp(mask_tokens_out[:, 0, :])  # shape => [B, mlp_out] (mlp_out=3*(C'))
        # reshape => [B, 3, (C')]
        cprime = upscaled_embedding.shape[1]  # = transformer_dim // 8
        # we expect hyper_in.shape => [B, 3*cprime]
        hyper_in = hyper_in.view(b, 3, cprime)  # => [B, 3, C']

        # Next, we do: [B,3,C'] @ [B,C',Hup*Wup] => [B,3,Hup*Wup]
        # We'll flatten upscaled_embedding for matrix multiply
        # But we must do it for each batch
        upscaled_flat = upscaled_embedding.view(b, cprime, -1)  # => [B, C', h*w]
        # We'll do a batch multiply => we can do hyper_in @ upscaled_flat with bmm
        # but we need hyper_in => [B, 3, cprime], upscaled_flat => [B, cprime, h*w]
        # => output => [B, 3, h*w]
        masks_flat = torch.bmm(hyper_in, upscaled_flat)  # => [B, 3, h*w]
        # reshape => [B, 3, Hup, Wup]
        B_, Nch, hw = masks_flat.shape
        h2, w2 = upscaled_embedding.shape[-2:]
        masks = masks_flat.view(B_, Nch, h2, w2)

        # iou predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # For no object or dynamic logic, skip for brevity
        # Return shape => [B, 3, Hup, Wup] is the raw mask logits
        # The original code returns (all_mask_logits, iou_pred, mask_tokens_out, object_score_logits).
        # We'll do the same but skip object scores.
        object_score_logits = iou_pred.new_ones((bsz, 1)) * 10.0  # dummy if needed

        return masks, iou_pred, mask_tokens_out, object_score_logits
