import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import torchvision
import torchvision.datasets as dset
import torchvision.models as models




###### break resnet152 at one conv layer

model = torchvision.models.resnet152(pretrained=True)
model_s = list(model.children())

model_first = nn.Sequential(*(model_s[:6]))
model_second = nn.Sequential(model_s[6], model_s[7], model_s[8], nn.Flatten(), model_s[9])


torch.save(model_first, "model_first_part.pt")
torch.save(model_second, "model_second_part.pt")



###### break resnet152 at another conv layer


# model = torchvision.models.resnet152(pretrained=True)
# model_s = list(model.children())

# model_first = nn.Sequential(*(model_s[:7]))
# model_second = nn.Sequential(model_s[7], model_s[8], nn.Flatten(), model_s[9])


# torch.save(model_first, "model_first_part.pt")
# torch.save(model_second, "model_second_part.pt")



###### break resnet152 at final linear layer

# model = torchvision.models.resnet152(pretrained=True)
# model_s = list(model.children())

# model_first = nn.Sequential(*(model_s[:9]), nn.Flatten())
# model_second = nn.Sequential(model_s[9])


# torch.save(model_first, "model_first_part.pt")
# torch.save(model_second, "model_second_part.pt")


###### break vision transformer at final layer

# model_first = torchvision.models.vit_b_16(pretrained=True)
# model_first.heads = nn.Sequential()
# model_second = list(torchvision.models.vit_b_16(pretrained=True).children())
# model_second = nn.Sequential(model_second[-1])

# torch.save(model_first, "model_first_part.pt")
# torch.save(model_second, "model_second_part.pt")