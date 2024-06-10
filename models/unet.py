import torch
import torch.nn as nn 

def get_time_embedding(time_steps, time_emb_dim):
    assert time_emb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    freq = 10000 ** (torch.arange(start=0, end=time_emb_dim // 2, dtype=torch.float32, device=time_steps.device) / (time_emb_dim // 2))
    time_emb = time_steps.unsqueeze(-1).repeat(1, time_emb_dim // 2) / freq
    time_emb =  torch.cat([torch.sin(time_emb), torch.cos(time_emb)], dim=-1)
    return time_emb


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, down_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.resnet_block_first = nn.ModuleList([
                                    nn.Sequential(
                                        nn.GroupNorm(8, in_channels if i==0 else out_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(in_channels if i==0 else out_channels, out_channels, 3, stride=1, padding=1),
                                    )
                                    for i in range(num_layers)
                                ])
        
        self.t_emb_layers = nn.ModuleList([
                                nn.Sequential(
                                    nn.SiLU(),
                                    nn.Linear(time_emb_dim, out_channels),
                                )
                                for _ in range(num_layers)
                            ])

        self.resnet_block_second = nn.ModuleList([
                                    nn.Sequential(
                                        nn.GroupNorm(8, out_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
                                    ) 
                                    for _ in range(num_layers)
                                ])
        
        self.attention_norms = nn.ModuleList([
                                nn.GroupNorm(8, out_channels)
                                for _ in range(num_layers)
                            ])

        self.attention_blocks = nn.ModuleList([
                                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                                for _ in range(num_layers)
                            ])
        
        self.residue_resnet_conv = nn.ModuleList([
                                    nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                                    for i in range(num_layers)
                                ])
        
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if down_sample else nn.Identity()

    def forward(self, x, time_emb):
        out = x

        for i in range(self.num_layers):
            resnet_in = out
            out = self.resnet_block_first[i](out)
            # print(time_emb[:, :, None, None].shape)
            out = out + self.t_emb_layers[i](time_emb)[:, :, None, None]
            out = self.resnet_block_second[i](out)
            out = out + self.residue_resnet_conv[i](resnet_in)

            batch_size, channels, h, w = out.shape
            in_attention = out.reshape(batch_size, channels, h*w)
            in_attention = self.attention_norms[i](in_attention)
            in_attention = in_attention.transpose(1, 2)
            out_attention, _ = self.attention_blocks[i](in_attention, in_attention, in_attention)
            out_attention = out_attention.transpose(1, 2).reshape(batch_size, channels, h, w)
            # print(out.shape, out_attention.shape)
            out = out + out_attention

        out = self.down_sample_conv(out)
            
        return out
    

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.resnet_block_first = nn.ModuleList([
                                    nn.Sequential(
                                        nn.GroupNorm(8, in_channels if i==0 else out_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(in_channels if i==0 else out_channels, out_channels, 3, stride=1, padding=1),
                                    )
                                    for i in range(num_layers+1)
                                ])
        
        self.t_emb_layers = nn.ModuleList([
                                nn.Sequential(
                                    nn.SiLU(),
                                    nn.Linear(time_emb_dim, out_channels),
                                )
                                for i in range(num_layers+1)
                            ])

        self.resnet_block_second = nn.ModuleList([
                                    nn.Sequential(
                                        nn.GroupNorm(8, out_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
                                    ) 
                                    for i in range(num_layers+1)
                                ])
        
        self.attention_norms = nn.ModuleList([
                                nn.GroupNorm(8, out_channels)
                                for i in range(num_layers)
                            ])

        self.attention_blocks = nn.ModuleList([
                                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                                for i in range(num_layers)
                            ])
        
        self.residue_resnet_conv = nn.ModuleList([
                                    nn.Sequential(
                                        nn.GroupNorm(8, in_channels if i==0 else out_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1),
                                    )
                                    for i in range(num_layers+1)
                                ])

    def forward(self, x, time_emb):
        out = x
        resnet_in = out
        
        out = self.resnet_block_first[0](out)
        out = out + self.t_emb_layers[0](time_emb)[:, :, None, None]
        out = self.resnet_block_second[0](out)
        out = out + self.residue_resnet_conv[0](resnet_in)

        for i in range(self.num_layers):
            batch_size, channels, h, w = out.shape
            in_attention = out.reshape(batch_size, channels, h*w)
            in_attention = self.attention_norms[i](in_attention)
            in_attention = in_attention.transpose(1, 2)
            out_attention, _ = self.attention_blocks[i](in_attention, in_attention, in_attention)
            out_attention = out_attention.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attention

            resnet_in = out
            out = self.resnet_block_first[i+1](out)
            out = out + self.t_emb_layers[i+1](time_emb)[:, :, None, None]
            out = self.resnet_block_second[i+1](out)
            out = out + self.residue_resnet_conv[i+1](resnet_in)
            
        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample

        self.resnet_block_first = nn.ModuleList([
                                    nn.Sequential(
                                        nn.GroupNorm(8, in_channels if i==0 else out_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(in_channels if i==0 else out_channels, out_channels, 3, stride=1, padding=1),
                                    )
                                    for i in range(num_layers)
                                ])
        
        self.t_emb_layers = nn.ModuleList([
                                nn.Sequential(
                                    nn.SiLU(),
                                    nn.Linear(time_emb_dim, out_channels),
                                )
                                for i in range(num_layers)
                            ])

        self.resnet_block_second = nn.ModuleList([
                                    nn.Sequential(
                                        nn.GroupNorm(8, out_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
                                    ) 
                                    for i in range(num_layers)
                                ])
        
        self.attention_norms = nn.ModuleList([
                                nn.GroupNorm(8, out_channels)
                                for i in range(num_layers)
                            ])

        self.attention_blocks = nn.ModuleList([
                                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                                for i in range(num_layers)
                            ])
        
        self.residue_resnet_conv = nn.ModuleList([
                                    nn.Sequential(
                                        nn.GroupNorm(8, in_channels if i==0 else out_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1),
                                    )
                                    for i in range(num_layers)
                                ])
        
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()
        

    def forward(self, x, down_out, time_emb):
        x = self.up_sample_conv(x)
        out = torch.cat([x, down_out], dim=1)
        
        for i in range(self.num_layers):
            resnet_in = out
            out = self.resnet_block_first[i](out)
            out = out + self.t_emb_layers[i](time_emb)[:, :, None, None]
            out = self.resnet_block_second[i](out)
            out = out + self.residue_resnet_conv[i](resnet_in)

            batch_size, channels, h, w = out.shape
            in_attention = out.reshape(batch_size, channels, h*w)
            in_attention = self.attention_norms[i](in_attention)
            in_attention = in_attention.transpose(1, 2)
            out_attention, _ = self.attention_blocks[i](in_attention, in_attention, in_attention)
            out_attention = out_attention.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attention
            
        return out

class Unet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.time_emb_dim = model_config.time_emb_dim
        self.down_channels = model_config.down_channels
        self.down_sample = model_config.down_sample
        self.mid_channels = model_config.mid_channels
        self.num_heads = model_config.num_heads
        image_channels = model_config.image_channels

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        self.conv_in = nn.Conv2d(image_channels, self.down_channels[0], 3, padding=1)

        for i in range(len(self.down_channels)-1):
            self.down_blocks.append(DownBlock(self.down_channels[i], self.down_channels[i+1], 128, down_sample=self.down_sample[i]))

        for i in range(len(self.mid_channels)-1):
            self.mid_blocks.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], 128))

        for i in reversed(range(len(self.down_channels)-1)):
            self.up_blocks.append(UpBlock(2*self.down_channels[i], self.down_channels[i-1] if i else 16, 128, up_sample=self.down_sample[i]))

        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, image_channels, 3, padding=1)
        

    def forward(self, x, t):

        out = self.conv_in(x)
        time_emb = get_time_embedding(t, self.time_emb_dim)
        down_out_list = []

        for i in range(len(self.down_channels)-1):
            down_out_list.append(out)
            out = self.down_blocks[i](out, time_emb)
            
        for i in range(len(self.mid_channels)-1):
            out = self.mid_blocks[i](out, time_emb)

        for i in range(len(self.down_channels)-1):
            down_out = down_out_list.pop()
            out = self.up_blocks[i](out, down_out, time_emb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out

