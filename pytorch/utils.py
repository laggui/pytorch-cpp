import torch

def try_load(load_path):
    state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)
    if next(iter(state_dict.keys()))[0:6] == 'module':
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict