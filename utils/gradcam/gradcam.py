
from torchvision.models import *
from visualisation.core.utils import device
from efficientnet_pytorch import EfficientNet
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
# from utils import *
import PIL.Image
import cv2

from visualisation.core.utils import device
from visualisation.core.utils import image_net_postprocessing

from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from visualisation.core import *
from visualisation.core.utils import image_net_preprocessing
from visualisation.core.utils import tensor2img
# from utils import tensor2img

from IPython.display import Image
from matplotlib.animation import FuncAnimation
from collections import OrderedDict


def efficientnet(model_name='',**kwargs):
    model = EfficientNet.from_pretrained("efficientnet-b0",num_classes=41)
    model.load_state_dict(torch.load("../../classification/efficientnet_models/41.pt"))
    return model.eval().to(device)

def efficientnet2(model_name='',**kwargs):
    model = EfficientNet.from_pretrained("efficientnet-b0",num_classes=41)
    model.load_state_dict(torch.load("../../classification/efficientnet_models/dp05_dc05.pt"))
    return model.eval().to(device)

max_img = 3
path = './test_data'
interesting_categories = ['091.test']

images = []
for category_name in interesting_categories:
    image_paths = glob.glob(f'{path}/{category_name}/*')
    category_images = list(map(lambda x: PIL.Image.open(x), image_paths[:max_img]))
    images.extend(category_images)

inputs  = [Compose([Resize((224,224)), ToTensor(), image_net_preprocessing])(x).unsqueeze(0) for x in images]  # add 1 dim for batch
inputs = [i.to(device) for i in inputs]

model_outs = OrderedDict()
model_instances = [
                  lambda pretrained:efficientnet(model_name='efficientnet-b0'),
                  lambda pretrained:efficientnet2(model_name='efficientnet-b3')]


model_names = [m.__name__ for m in model_instances]
model_names[-2],model_names[-1] = 'aihub_sujong','aihub_mingeon'

images = list(map(lambda x: cv2.resize(np.array(x),(224,224)),images)) # resize i/p img
# print(images)
for name, model in zip(model_names, model_instances):
	print(name)
	module = model(pretrained=True).to(device)
	module.eval()
	print(type(module))

	vis = GradCam(module, device)

	model_outs[name] = list(map(lambda x: tensor2img(vis(x, None, postprocessing=image_net_postprocessing)[0]), inputs))
	del module
	torch.cuda.empty_cache()

# create a figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 20))
axes = [ax2, ax3]


def update(frame):
	all_ax = []
	ax1.set_yticklabels([])
	ax1.set_xticklabels([])
	ax1.text(1, 1, 'Orig. Im', color="white", ha="left", va="top", fontsize=30)
	all_ax.append(ax1.imshow(images[frame]))
	for i, (ax, name) in enumerate(zip(axes, model_outs.keys())):
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		ax.text(1, 1, name, color="white", ha="left", va="top", fontsize=20)
		all_ax.append(ax.imshow(model_outs[name][frame], animated=True))

	return all_ax


ani = FuncAnimation(fig, update, frames=range(len(images)), interval=1000, blit=True)
model_names = [m.__name__ for m in model_instances]
model_names = ', '.join(model_names)
fig.tight_layout()
ani.save('./compare_arch.gif', writer='imagemagick')

Image('./compare_arch.gif')