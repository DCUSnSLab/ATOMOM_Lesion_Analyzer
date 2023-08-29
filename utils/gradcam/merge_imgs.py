import os
from PIL import Image, ImageDraw, ImageFont
import glob
import cv2
import natsort
import pandas as pd
import shutil
def merge_images(subdirs, merge_dir,origin_dir):
	try:
		shutil.rmtree(merge_dir)
	except FileNotFoundError:
		pass
	os.makedirs(merge_dir, exist_ok=True)

	image_names = os.listdir(subdirs[0])
	# print(image_names)
	# print(natsort.natsorted(subdirs))
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.5
	font_color = (255, 255, 255)  # White color
	text_position = (10, 30)  # Top left corner coordinates
	correct_color = (0, 255, 0)  # Green color for correct predictions
	incorrect_color = (0, 0, 255)  # Red color for incorrect predictions
	padding = 5
	for cur_img_name in image_names:
		merge_img = cv2.resize(cv2.imread(os.path.join(origin_dir, cur_img_name)),(224,224))
		merge_img= cv2.copyMakeBorder(merge_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
		for i, subdir in enumerate(subdirs):
			print(subdir)

			df = pd.read_csv(os.path.join(os.path.dirname(subdir),os.path.basename(subdir))+'.csv')
			row = df[df['img_names'] == cur_img_name.split('.')[0]]
			GT = row['Ground Truth'].values[0]
			if(GT == 'miso' or GT == 'sujong'):
				GT = 'normal'
			prediction = row['predict'].values[0]
			confidence = row['confidence'].values[0]
			# print(GT,prediction,confidence)
			# exit()

			text = os.path.basename(subdir)
			text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
			text_position = (10, text_size[1] + 10)
			cur_img = cv2.imread(os.path.join(subdir, cur_img_name))
			cv2.putText(cur_img, text, text_position, font, font_scale, font_color, 2)

			# prediction = 'atopy'  # Replace this with your actual prediction
			# confidence = 0.95  # Replace this with your actual confidence
			text_to_display = f'{prediction}'
			text_size_details = cv2.getTextSize(text_to_display, font, font_scale, 2)[0]
			text_position_details = (10, text_size_details[1] * 2 + 20)  # Adjust Y-coordinate
			cv2.putText(cur_img, text_to_display, text_position_details, font, font_scale, font_color, 2)

			text_to_display = f'{confidence:.2f}'
			text_size_details = cv2.getTextSize(text_to_display, font, font_scale, 2)[0]
			text_position_details = (10, text_size_details[1] * 2 + 40)  # Adjust Y-coordinate
			cv2.putText(cur_img, text_to_display, text_position_details, font, font_scale, font_color, 2)


			# rectangle_color = correct_color if GT == prediction else incorrect_color
			rectangle_color = incorrect_color if (GT == 'atopy' and prediction != 'atopy') or \
			                                     (GT != 'atopy' and prediction == 'atopy') else correct_color

			rectangle_thickness = 8
			cv2.rectangle(cur_img, (0, 0), (cur_img.shape[1], cur_img.shape[0]), rectangle_color, rectangle_thickness)
			cur_img = cv2.copyMakeBorder(cur_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT,value=(0, 0, 0))
			merge_img = cv2.hconcat([merge_img,cur_img])
		cv2.imwrite(os.path.join(merge_dir,cur_img_name),merge_img)


def concatenate_images_vertically(image_dir, output_path,concat_num):
	count = 0
	counting_img = 0
	merge_img = None
	try:
		shutil.rmtree(output_path)
	except FileNotFoundError:
		pass
	os.makedirs(output_path, exist_ok=True)
	img_pathes = natsort.natsorted(os.listdir(image_dir))
	for idx, img_path in enumerate(img_pathes):
		img_path = os.path.join(merge_dir,img_path)
		if(counting_img==0):
			merge_img = cv2.imread(img_path)
			counting_img +=1
			continue
		cur_img = cv2.imread(img_path)
		merge_img = cv2.vconcat([merge_img, cur_img])
		counting_img +=1
		if(counting_img>=concat_num or idx+1>=len(img_pathes)):
			counting_img = 0
			cv2.imwrite(os.path.join(output_path,"merged_"+str(count)+".jpg"),merge_img)
			count+=1

	# for subdir in subdirs:
	# 	print(subdir)
	# 	img_name_list = []
	# 	images = []
	# 	for img_path in glob.glob(os.path.join(subdir, '*.jpg')):
	# 		cv2.hconcat
	# 	# images = [Image.open(img_path) ]
	# 	# # print(images[0])
	# 	# # exit()
	# 	merged_image = Image.new('RGB', (sum(img.width for img in images), images[0].height))
	#
	# 	x_offset = 0
	# 	for img in images:
	# 		merged_image.paste(img, (x_offset, 0))
	# 		x_offset += img.width
	#
	# 	# draw = ImageDraw.Draw(merged_image)
	# 	# font = ImageFont.truetype("arial.ttf", 30)  # Change the font and size as needed
	# 	# draw.text((10, 10), subdir, fill=(255, 255, 255), font=font)
	# 	print("merge_dir",merge_dir)
	# 	merge_save_path = os.path.join(merge_dir, f'merged_{os.path.basename(subdir)}.jpg')
	# 	print("merge_save_path",merge_save_path)
	# 	merged_image.save(merge_save_path)


if __name__ == '__main__':
	path = '~/ATOMOM_Lesion_Analyzer/tmp'
	path = os.path.expanduser(path)
	subdirs = []
	origin_dir = os.path.join(path,"original")
	concat_num = 10
	output_path = os.path.join(path, 'merge_vertically_' + str(concat_num))
	for item in os.listdir(path):
		cur_path = path + '/' + item
		#    path = os.path.join(root_dir, item)
		# print(cur_path)
		if os.path.isdir(cur_path) and os.path.basename(cur_path)!='merge' and os.path.basename(cur_path)!='original'and os.path.basename(cur_path)!='merge_vertically_' + str(concat_num):
			subdirs.append(os.path.join(path,cur_path))
	subdirs = natsort.natsorted(subdirs)
	# print(subdirs)
	# subdirs = ['B', 'C', 'D']  # List of subdirectories to merge images from
	merge_dir = os.path.join(path, 'merge')  # Path to the merge directory within A
	# print(merge_dir)
	merge_images(subdirs, merge_dir,origin_dir)


	print(merge_dir)
	concatenate_images_vertically(image_dir=merge_dir,output_path=output_path,concat_num=concat_num)

