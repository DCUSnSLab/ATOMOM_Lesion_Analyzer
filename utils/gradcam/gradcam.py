from torchvision.models import *
from efficientnet_pytorch import EfficientNet
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import PIL.Image
import cv2
from visualisation.core.utils import device
from visualisation.core.utils import image_net_postprocessing
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from visualisation.core import *
from visualisation.core.utils import image_net_preprocessing
from visualisation.core.utils import tensor2img
from IPython.display import Image
from matplotlib.animation import FuncAnimation
from collections import OrderedDict
import os
from tqdm import tqdm

# def efficientnet(model_name='', **kwargs):
# 	model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=41)
# 	model.load_state_dict(torch.load("../../classification/efficientnet_models/second_learning_result_efficientNet_b0_su_class_41.pt"))
# 	return model.eval().to(device)
#
#
# def efficientnet2(model_name='', **kwargs):
# 	model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=41)
# 	model.load_state_dict(torch.load("../../classification/efficientnet_models/second_learning_result_efficientNet_b0_mingeon_class_41.pt"))
# 	return model.eval().to(device)
def efficientnet(model_name='', pretrained_path='',num_class='', **kwargs):
	model = EfficientNet.from_pretrained(model_name, num_classes=int(num_class))
	model.load_state_dict(torch.load(pretrained_path))
	return model.eval().to(device), model_name

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


def process_image(x,vis):
	visualization = vis(x, None, postprocessing=image_net_postprocessing)[0]
	return tensor2img(visualization)

class Grad:
	def __init__(self,img_path=None,vis_max_img=112,is_standalone=False,model=None):
		self.is_standalone = is_standalone
		if self.is_standalone:
			assert img_path is not None and model is not None, "img_path and model_path must not be None when is_standalone is True"
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			self.image_paths = glob.glob(f'{img_path}/*')
			# print(img_path)
			# print(self.image_paths)
			images = list(map(lambda x: PIL.Image.open(x), self.image_paths[:vis_max_img]))
			inputs = [Compose([Resize((224, 224)), ToTensor(), image_net_preprocessing])(x).unsqueeze(0) for x in
			          images]  # add 1 dim for batch
			self.inputs = [i.to(self.device) for i in inputs]
			self.images = list(map(lambda x: cv2.resize(np.array(x), (224, 224)), images))
		self.model = model
	def __process_image(self, x,vis):
		visualization = vis(x, None, postprocessing=image_net_postprocessing)[0]
		return tensor2img(visualization)

	def __save_processed_images(self, model_outs, save_path):
		os.makedirs(save_path, exist_ok=True)

		for idx, (img, img_path) in enumerate(zip(model_outs, self.image_paths)):
			# print("dsdf")
			result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			img_filename = os.path.basename(img_path)
			save_file = os.path.join(save_path, f"{img_filename}")
			# print(save_file)
			# cv2.imshow("result",result)
			# cv2.waitKey(3000)
			# print(result.shape)
			cv2.imwrite(save_file, result*255)
	def save_src_images(self,save_path):
		save_path = os.path.expanduser(save_path)
		os.makedirs(save_path, exist_ok=True)
		for idx, (img, img_path) in enumerate(zip(self.images, self.image_paths)):
			result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			img_filename = os.path.basename(img_path)
			save_file = os.path.join(save_path, f"{img_filename}")
			cv2.imwrite(save_file, result)



	def inference(self,model=None,vis=True,save_path = '~/ATOMOM_Lesion_Analyzer/exp_results/grad',model_name = None):

		if self.is_standalone:
			inputs = self.inputs
		# if(model_name == None):
		# 	model_name = model.__name__
		save_path = os.path.expanduser(save_path)
		os.makedirs(save_path, exist_ok=True)
		vis = GradCam(self.model, device)
		# print(type(vis))
		model_outs = [self.__process_image(x,vis) for x in self.inputs]
		self.__save_processed_images(model_outs=model_outs,save_path=save_path)


if __name__ == '__main__':
	max_img = 110
	path = '~/ATOMOM_Lesion_Analyzer/test_data/atomom_test_images_samples'
	path = os.path.expanduser(path)
	save_path = '~/ATOMOM_Lesion_Analyzer/exp_results/'
	save_path = os.path.expanduser(save_path)
	model_path = '~/ATOMOM_Lesion_Analyzer/classification/efficientnet_models'
	model_path = os.path.expanduser(model_path)
	src_save_path = '~/ATOMOM_Lesion_Analyzer/classification/efficientnet_models/resized_src'

	model_list = [file for file in os.listdir(model_path) if file.endswith(".pt")]
	for i in tqdm(range(len(model_list))):
		cur_model_path = model_list[i]
		model_name = cur_model_path.split('_')
		num_class = model_name[-1].split('.')[0]
		model_name = model_name[3].lower() + '-' + model_name[4]
		save_dir_name = cur_model_path.split('.')[0]
		# print(save_dir_name)
		cur_model_path = os.path.join(model_path,cur_model_path)
		cur_save_path = os.path.join(save_path,save_dir_name)

		model, model_name = efficientnet(model_name=model_name, num_class=num_class, pretrained_path=cur_model_path)
		grad = Grad(img_path=path, vis_max_img=max_img, is_standalone=True, model=model)
		# grad.inference(save_path= cur_save_path)
		grad.save_src_images(src_save_path)

	# grad = Grad(img_path=path, vis_max_img=max_img, is_standalone=True, model=model)
	# grad.inference()
