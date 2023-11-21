from ultralytics import YOLO
import numpy as np
import cv2

from setproctitle import *
import os
from copy import deepcopy

class Yolo_Seg:
    def __init__(self,config):

        self.Config = config
        self.model = YOLO(config.model_path)
        self.Config.model_names = self.model.names

    def __get_image_paths(self, image_path):
        if (os.path.isfile(image_path)):
            self.Config.file_paths = [image_path]
        elif (os.path.isdir(image_path)):
            file_paths = [x for x in os.listdir(image_path)]
            for i, file in enumerate(file_paths):
                if ('.png' in file or '.jpg' in file):
                    pass
                else:
                    file_paths[i] = None
            temp_indexes = list(np.where(np.array(file_paths) != None)[0])
            file_paths = [os.path.join(image_path, file_paths[x]) for x in temp_indexes]
            if (len(file_paths) > 0):
                self.Config.file_paths = file_paths
            else:
                raise Exception("No valid image files found, please check dir")
        else:
            raise Exception("image_path is not dir or valid image, please check image_path")
    def __draw_boexes(self,image, boxes):
        cropped_images = [None for _ in range(len(boxes.boxes))]
        confidences = [None for _ in range(len(boxes.boxes))]
        for i, box in enumerate(boxes.boxes):
            lw = max(round(sum(image.shape) / 2 * 0.003), 2)
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            # print(box.cpu().numpy())
            # print(box[4].cpu().numpy(),box[5].cpu().numpy())
            confidences[i] = box[4].cpu().numpy()
            # print("*"*50)
            # confidence[i]
            cropped_images[i] = self.copied_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if self.Config.bbox:
                cv2.rectangle(image, p1, p2, (0, 0, 255), thickness=lw, lineType=cv2.LINE_AA)
            cur_class = self.Config.model_names[int(boxes.cls[i])]

            if self.Config.label:
                cur_label = '' + cur_class
                tf = max(lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(cur_label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(image, p1, p2, (0, 0, 255), -1, cv2.LINE_AA)  # filled
                cv2.putText(image, cur_label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3,
                            (255, 255, 255),
                            thickness=tf, lineType=cv2.LINE_AA)
        return image, cropped_images, confidences

    def __draw_segmentation2(self, image, segmentations, alpha=0.8):
        for segmentation in segmentations:
            segmentation = segmentation.cpu().numpy()
            segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]))
            color = np.array([0, 0, 255], dtype='uint8')
            masked_img = np.where(segmentation[..., None], color, image)
            image = cv2.addWeighted(src1=image, alpha=alpha, src2=masked_img, beta=0.2, gamma=0)
        # if (True):
        #     cv2.imshow("image", image)
        #     cv2.waitKey(30000)
        return image

    def __draw_segmentation(self, image, segmentations, alpha=0.8):
        if len(segmentations) == 0:
            return image

        segmentations = [cv2.resize(seg.cpu().numpy(), (image.shape[1], image.shape[0])) for seg in segmentations]
        # Stacking segmentations to shape (h, w, num_masks)
        stacked_masks = np.stack(segmentations, axis=-1)

        # Create a mask where any segmentation is active
        combined_mask = np.any(stacked_masks, axis=-1)

        # Create color overlay
        color_overlay = np.zeros(image.shape, dtype=image.dtype)
        color_overlay[combined_mask] = [0, 0, 255]

        # Blend color_overlay with the original image
        blended = cv2.addWeighted(src1=image, alpha=1, src2=color_overlay, beta=0.2, gamma=0)
        return blended


    def __get_results(self, image_info,visualize):
        image = None
        if isinstance(image_info, str):
            image = cv2.imread(image_info)
        else:
            image = image_info
        self.copied_image = deepcopy(image)
        # print("self.Config.device",self.Config.device)
        results = self.model.predict(image, device=self.Config.device, verbose=self.Config.verbose)
        # print(results[0].boxes.boxes)
        length = len(results[0].boxes.cpu().numpy())
        cropped_images = image
        confidences = []
        if (length <= 0):
            # print('\033[31m' + "object is not detected!!!", "image_path: " + cur_image_path + '\033[0m')
            # return image, [cropped_images], False, [0.0]
            # print("여기로 오네")
            return image, [], False, [0.0]
        # raise Exception('\033[31m' + "object is not detected!!!", "image_path: " + image_path + '\033[0m')
        else:
            # print("여기로 오네2")
            for i, result in enumerate(results):
                image,cropped,confidences = self.__draw_boexes(image=image, boxes=result.boxes)
                cropped_images = cropped
                if(visualize):
                    image = self.__draw_segmentation(image, segmentations=result.masks.masks)
                # cv2.imshow("image")
                # cv2.waitKey(0)
        return image, cropped_images, True, confidences


    def inference(self, image_info,save_path=False,visualize = False):
        if isinstance(image_info, str):
            self.__get_image_paths(image_info)
        elif isinstance(image_info, np.ndarray):
            self.Config.file_paths = [image_info]
        assert len(self.Config.file_paths)>0, "경로 또는 ndarray를 입력하세요"

        inference_results = [None for _ in range(len(self.Config.file_paths))]

        for i, image_info in enumerate(self.Config.file_paths):
            image, cropped_images,status, confidences = self.__get_results(image_info=image_info,visualize=visualize)
            inference_results[i] = (image,cropped_images,status, confidences)

            if (save_path):
                raise NotImplementedError("save function is not implemented ㅜㅜ")
            if (self.Config.display):
                cv2.imshow("image", image)
                cv2.waitKey(1)
        return inference_results





if __name__ == '__main__':
    setproctitle('yolo test')
    class Config_yolo():
        model_path = "segmentation/yolo_models/best_n.pt"
        model_names = None
        display = True
        save_path = None
        verbose = False
        device = 1
        label = False
        bbox = False
        segmentation = True
        file_paths = None

    image_path = "/home/dgdgksj/ATOMOM_Lesion_Analyzer/test_data/atomom_test_images_samples/atopy_video_0457.jpg"
    # image_path = "/home/dgdgksj/skin_lesion/ultralytics/atomom_test_images/"
    yolo_seg = Yolo_Seg(config=Config_yolo)





    inference_results = yolo_seg.inference(image_info=image_path,visualize=True)
    for i, data in enumerate(inference_results):
        for index,cr in enumerate(data[1]):
            cv2.imshow(str(index),cr)
            cv2.imwrite(str(index)+".jpg",cr)
        cv2.imwrite("sdf.jpg", data[0])
        cv2.imshow("results",data[0])
        cv2.waitKey(3000)
        cv2.destroyAllWindows()