import torch
import torch.nn as nn
import torch.nn.functional as F

from .NMT import NMA, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from utils.utils import calc_diffusion_step_embedding


def swish(x):
    return x * torch.sigmoid(x)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.0, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm2(x + y)
        return y


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DDTM(nn.Module):
    def __init__(self, enc_in, c_out, d_model=512, n_heads=8, d_ff=512,
                 diffusion_step_embed_dim_in=256, diffusion_step_embed_dim_mid=512,
                 diffusion_step_embed_dim_out=512, dropout=0.1, mask_scale=0, activation='relu', output_attention=True):
        super(DDTM, self).__init__()
        self.output_attention = output_attention
        self.mask_scale = mask_scale
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in
        self.d_ff = d_ff
        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.kernel_size = 3
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(NMA(mask_scale=self.mask_scale, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        self.fc_t3 = nn.Linear(diffusion_step_embed_dim_out, d_ff)
        self.projection = nn.Conv1d(d_model, c_out, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)
        self.cond_conv = nn.Conv1d(enc_in, d_model, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)

    def forward(self, x):
        noise, conditional, mask, diffusion_steps = x
        x = noise.permute(0, 2, 1)

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))
        part_t = self.fc_t3(diffusion_step_embed)
        B, C = part_t.shape
        part_t = part_t.view([B, 1, C])
        enc_out = self.embedding(x) + part_t

        enc_out = self.encoder(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.projection(enc_out)

        return enc_out


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),  # 输入卷积层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),  # 编码卷积层
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1),  # 解码卷积层
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, input_dim, kernel_size=3, padding=1)  # 输出卷积层
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
