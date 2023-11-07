from IPython.core.display_functions import display
from algorithm.utils.data.dataimage import DataImage
import torch
from torch.utils.data import DataLoader
from PIL import Image


batch_size = 4


trainset = DataImage(data_path="Input/dataset", split="Train", name="Toy Dataset", transform=None, normalize=True, resize=True, height='auto', width='auto', mean='auto', std='auto')

display(trainset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

display(trainloader)

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

display(trainset.data[4])

display(dataiter)



file = trainset.data[2]
img = Image.open(file)
try:
    lbl = trainset.labels[2]
except TypeError:
    lbl = None
print(img)##########
if trainset.transform is not None:
    print(trainset.transform)
    trainset = 4
    img = trainset.transform(img)
print(img)###########

