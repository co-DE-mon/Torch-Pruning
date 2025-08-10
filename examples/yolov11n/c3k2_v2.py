from ultralytics.nn.modules import Detect, C3k2, Conv, Bottleneck, C2PSA, C2f, A2C2f
import torch
import torch.nn as nn

def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

# # ---------------------- C3k2_v2 (Simple) ----------------------
class c3k2_v2_simple(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# ---------------------- C3k_v2 (-------) ----------------------
class c3k_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e) 
        # self.cv0 = Conv(c1, self.c, 1, 1) 
        self.cv1 = Conv(c1, self.c, 1, 1) 
        self.cv2 = Conv(c1, self.c, 1, 1) 
        self.cv3 = Conv((1+n) * self.c, c2, 1, 1)  # final projection
        self.m = nn.Sequential(*[Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(2*n)])
    
    def forward(self, x):
        y = [self.cv1(x)]
        y.append(self.m(self.cv2(x)))
        return self.cv3(torch.cat(y, 1))

# ---------------------- C3k2_v2(Complex) ----------------------
class c3k2_v2_complex(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, c2, 1, 1)  # learnable shortcut path
        self.cv1 = Conv(c1, c2, 1, 1)  # main feature path
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList([c3k_v2(self.c, self.c, n=2, shortcut=shortcut, g=g, e=e) for _ in range(n)])
    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)  # Static iteration over ModuleList
        return self.cv2(torch.cat(y, 1))


def transfer_weights_c3k2(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)

def replace_c3k2_with_c3k2_v2(module):
    """
    Replace C3k2 blocks with appropriate C3k2_v2 variants based on internal structure
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, C3k2):
            shortcut = infer_shortcut(child_module.m[0])
            # Determine if it's simple or complex based on internal structure
            has_c3k_subblock= any('C3k' in str(type(m).__name__) for m in module.m) if hasattr(module, 'm') else False

            if has_c3k_subblock:
                # Complex C3k2 block with C3k sub-blocks
                print(f"Replacing complex C3k2 block: {name}")
                c3k2_v2 = c3k2_v2_complex(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
                transfer_weights_c3k2(child_module, c3k2_v2)
                setattr(module, name, c3k2_v2)
            else:
                c3k2_v2 = c3k2_v2_simple(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
                transfer_weights_c3k2(child_module, c3k2_v2)
                setattr(module, name, c3k2_v2)
            
        else:
            # Recursively process child modules
            replace_c3k2_with_c3k2_v2(child_module)
