import json

import torch

from waveglow.glow import WaveGlow
configname = "config.json"

with open(configname) as f:
    data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]

model_name = "waveglow_256channels.pt"

waveglow = torch.load(model_name)['model']
print("testing")

model = WaveGlow(**waveglow_config).cuda()

torch.save({'model': waveglow,
            'iteration': 0,
            'optimizer': torch.optim.Adam(model.parameters(), lr=1e-4).state_dict(),
            'learning_rate': 1e-4}, "testmodel.pt")