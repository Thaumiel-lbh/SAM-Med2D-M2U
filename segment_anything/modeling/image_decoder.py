import torch
import torch.nn as nn
from .image_encoder import Block
from typing import Optional, Tuple, Type

class ImageDecoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        in_chans: int = 3,
        patch_size: int = 16,
        embed_dim: int = 256,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 12,
        decoder_num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Encoder patch embedding dimension.
            decoder_dim (int): Patch embedding dimension.
            decoder_depth (int): Depth of ViT.
            decoder_num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, 
                  num_heads=decoder_num_heads, 
                  mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, 
                  norm_layer=norm_layer,
                  act_layer = act_layer, 
                  use_rel_pos=use_rel_pos, 
                  rel_pos_zero_init=rel_pos_zero_init,
                  window_size=window_size if i not in global_attn_indexes else 0,
                  input_size=(img_size // patch_size, img_size // patch_size))
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
    

    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h, w, p**2 * 3))
        return x
    
    def forward_decoder(self, x):
        x = x.permute(0, 2, 3, 1)
        
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        return x
    
    def forward_loss(self, imgs, pred):
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        return loss.sum()
    
    def forward(self, imgs, x):
        pred = self.forward_decoder(x)
        loss = self.forward_loss(imgs, pred)
        
        return pred, loss