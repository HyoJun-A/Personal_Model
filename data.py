#%%
#pip install pillow
from ast import increment_lineno
from logging.config import valid_ident
from PIL import Image

import matplotlib
from matplotlib import image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import pandas as pd

trans = transforms.Compose([transforms.Resize((100, 100)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
                            ])
trainset = torchvision.datasets.ImageFolder(root="D:/project/Personal_Model/personal",
                                            transform=trans)


trainset.__getitem__(18)


len(trainset)

classes =trainset.classes
classes

trainloader = DataLoader(trainset,
                         batch_size=16,
                         shuffle=False)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(labels)

#%%
def imshow(img):
    img= img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    
    print(np_img.shape)
    print((np.transpose(np_img, (1, 2, 0))).shape)

# %%
print(images.shape)

# %%
imshow(torchvision.utils.make_grid(images, nrow=4))
#%%
print(images.shape)

#%%
print((torchvision.utils.make_grid(images)).shape)
#%%
print("".join("%5s "%classes[labels[j]]for j in range(16)))


#%%
pd.Series(labels).value_counts()

#%%
plt.hist(labels)
plt.show()

#%%
np.random.seed(1234)
index_list = np.arange(0, len(labels))
valid_index = np.random.choice(index_list, size = 16, replace=False)


#%%
valid_img = images[valid_index]
valid_labels = labels[valid_index]

# %%
train_index = set(index_list) - set(valid_index)
images = images[list(train_index)]
labels = labels[list(train_index)]
# %%
pd.Series(valid_labels).value_counts()

#%%
print(valid_labels)
print(images)