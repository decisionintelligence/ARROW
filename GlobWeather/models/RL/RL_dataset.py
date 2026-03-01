import torch
from GlobWeather.data.iterative_dataset import get_data_given_path, get_out_path
from GlobWeather.models.RL.RL_utils import variables
import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob

def collate_fn_val(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, V, H, W

    out_dicts = [batch[i][1] for i in range(len(batch))]
    dayofyear_dicts = [batch[i][2] for i in range(len(batch))]
    hour_dicts = [batch[i][3] for i in range(len(batch))]
    list_lead_times = out_dicts[0].keys()
    out = {}
    dayofyear = {}
    hour = {}

    for lead_time in list_lead_times:
        out[lead_time] = torch.stack([out_dicts[i][lead_time] for i in range(len(batch))])
        dayofyear[lead_time] = torch.tensor([dayofyear_dicts[i][lead_time] for i in range(len(batch))])
        hour[lead_time] = torch.tensor([hour_dicts[i][lead_time] for i in range(len(batch))])
        
    variables = batch[0][4]
    year = batch[0][5]
    inp_file_idx = batch[0][6]
    
    return inp, out, dayofyear, hour, variables, year, inp_file_idx


class ERA5MultiLeadtimeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        transform,
        list_lead_times,
        data_freq=6,
    ):
        super().__init__()
        
        # lead times must be divisible by data_freq
        for l in list_lead_times:
            assert l % data_freq == 0

        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.list_lead_times = list_lead_times
        self.data_freq = data_freq
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = sorted(file_paths)
        max_lead_time = max(*list_lead_times) if len(list_lead_times) > 1 else list_lead_times[0]
        max_steps = max_lead_time // data_freq
        self.inp_file_paths = file_paths[:-max_steps] # the last few points do not have ground-truth
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.inp_file_paths)
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data = get_data_given_path(inp_path, self.variables)
        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        dict_out = {}
        dict_dayofyear = {}
        dict_hour = {}
        
        # get ground-truth paths at multiple lead times
        for lead_time in self.list_lead_times:
            out_path = get_out_path(self.root_dir, year, inp_file_idx, steps=lead_time // self.data_freq)
            dict_out[lead_time] = get_data_given_path(out_path, self.variables)
            dict_dayofyear[lead_time] = (inp_file_idx + lead_time // self.data_freq) // 4
            dict_hour[lead_time] = (inp_file_idx + lead_time // self.data_freq) % 4
            
        inp_data = torch.from_numpy(inp_data)
        dict_out = {lead_time: torch.from_numpy(out) for lead_time, out in dict_out.items()}
        
        return (
            self.transform(inp_data), # VxHxW
            {lead_time: self.transform(out) for lead_time, out in dict_out.items()},
            dict_dayofyear,
            dict_hour,
            self.variables,
            year,
            inp_file_idx,
        )


class Fine_tune_RLDataset:
    def __init__(
        self, 
        root_dir, 
        variables,
        inp_transform,
        out_transform_dict,
        list_intervals,
        data_freq
    ):
        self.root_dir = root_dir
        self.data_freq = data_freq
        self.variables = variables
        self.list_intervals = list_intervals
        self.inp_transform = inp_transform
        self.out_transform_dict = out_transform_dict

    def get_from_trajectory(self, year, time_idx, steps_trajectory):
        inp_data = []
        oup_data = []
        out_transform_mean = []
        out_transform_std = []
        for i, (y, idx) in enumerate(zip(year, time_idx)):
            last_out = get_data_given_path(os.path.join(self.root_dir, f"{y}_{idx:04}.h5"), variables)
            inp_data.append(torch.from_numpy(last_out))   # inp_data is also the last output
            diffs = []
            out_mean = []
            out_std = []
            for step in steps_trajectory[i]:
                out_path = get_out_path(self.root_dir, y, idx, step)
                out = get_data_given_path(out_path, variables)
                diff = out - last_out
                diff = torch.from_numpy(diff)
                if step != 0:
                    diffs.append(self.out_transform_dict[step*self.data_freq](diff))
                    out_mean.append(torch.from_numpy(self.out_transform_dict[step*self.data_freq].mean))
                    out_std.append(torch.from_numpy(self.out_transform_dict[step*self.data_freq].std))
                else:
                    diffs.append(diff)
                    out_mean.append(torch.from_numpy(self.out_transform_dict[6].mean))
                    out_std.append(torch.from_numpy(self.out_transform_dict[6].std))
            oup_data.append(torch.stack(diffs))
            out_transform_mean.append(torch.stack(out_mean))
            out_transform_std.append(torch.stack(out_std))

        inp_data = torch.stack(inp_data, dim=0)
        oup_data = torch.stack(oup_data, dim=0)
        out_transform_mean = torch.stack(out_transform_mean, dim=0)
        out_transform_std = torch.stack(out_transform_std, dim=0)

        interval_tensors = torch.from_numpy(steps_trajectory * self.data_freq).to(dtype=inp_data.dtype) / 10.0
        
        return (
            self.inp_transform(inp_data),
            oup_data,
            out_transform_mean,
            out_transform_std,
            interval_tensors,
        )