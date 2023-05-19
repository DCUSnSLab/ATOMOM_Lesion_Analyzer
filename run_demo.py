import os
import numpy as np
from setproctitle import *
import csv
import torch
import torch.onnx
import cv2
import natsort
from run_efficieintNet import Efficient_net
from run_yolo_seg import Yolo_Seg
from ultralytics import YOLO

def get_images_paths(image_path):
    if (os.path.isfile(image_path)):
        return [image_path]
    elif (os.path.isdir(image_path)):
        file_paths = [x for x in os.listdir(image_path)]
        file_paths = natsort.natsorted(file_paths)
        for i, file in enumerate(file_paths):
            if ('.png' in file or '.jpg' in file or '.JPG' in file):
                pass
            else:
                file_paths[i] = None
        temp_indexes = list(np.where(np.array(file_paths) != None)[0])
        file_paths = [os.path.join(image_path, file_paths[x]) for x in temp_indexes]
        if (len(file_paths) > 0):
            return file_paths
        else:
            raise Exception("No valid image files found, please check dir")
    else:
        raise Exception("image_path is not dir or valid image, please check image_path")

def ordinal_suffix(n):
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    if 10 < n < 20:
        return "%d%s" % (n, "th")
    return "%d%s" % (n, suffixes.get(n % 10, "th"))



class Skin_lesion:
    def __init__(self,ef_configs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cwd = os.getcwd()
        self.ef_configs = ef_configs
        self.num_ef_models = len(ef_configs)
        self.ef_models = self.__load_ef_net(ef_configs=self.ef_configs,device=self.device)

    def __load_ef_net(self,ef_configs,device):
        # os.chdir('./classification')
        models = [None for x in range(self.num_ef_models)]
        for i, config in enumerate(ef_configs):
            models[i] = Efficient_net(Config=config,device=device)
        # os.chdir(self.cwd)
        return models

    def __ef_inference(self, image_path):
        for i, model in enumerate(self.ef_models):
            ordinal = ordinal_suffix(i + 1)
            print(f"{ordinal} model:", model)
            print(model.inference(image_path=image_path))
        print('*'*50)
        # class_name, confidence = self.ef_41.inference(image_path=image_path)[0]
        # file_name = os.path.basename(image_path).split('.')[0]
        # return file_name,class_name,confidence
    def inference(self,image_path):
        self.__ef_inference(image_path)


if __name__ == '__main__':
    setproctitle('lesion')

    class Config_41():
        verbose = False
        topk = 41
        model_path = os.path.join(os.getcwd(),"classification/efficientnet_models/41.pt")
        model_name = 'efficientnet-b0'
        class_names = {
        "000": "normal_skin",
        "001": "atopy",
        "002": "prurigo",
        "003": "scar",
        "004": "psoriasis",
        "005": "varicella",
        "006": "nummular_eczema",
        "007": "ota_like_melanosis",
        "008": "becker_nevus",
        "009": "pyogenic_granuloma",
        "010": "acne",
        "011": "salmon_patches",
        "012": "dermatophytosis",
        "013": "wart",
        "014": "impetigo",
        "015": "vitiligo",
        "016": "ingrowing_nails",
        "017": "congenital_melanocytic_nevus",
        "018": "keloid",
        "019": "epidermal_cyst",
        "020": "insect_bite",
        "021": "molluscum_contagiosum",
        "022": "pityriasis_versicolor",
        "023": "melanonychia",
        "024": "alopecia_areata",
        "025": "epidermal_nevus",
        "026": "herpes_simplex",
        "027": "urticaria",
        "028": "nevus_depigmentosus",
        "029": "lichen_striatus",
        "030": "mongolian_spot_and_ectopic_mongolian_spot",
        "031": "capillary_malformation",
        "032": "pityriasis_lichenoides_chronica",
        "033": "infantile_hemangioma",
        "034": "mastocytoma",
        "035": "nevus_sebaceous",
        "036": "onychomycosis",
        "037": "milk_coffee_nevus",
        "038": "nail_dystrophy",
        "039": "melanocytic_nevus",
        "040": "juvenile_xanthogranuloma",
        }


    class Config_5():
        verbose = False
        topk = 5
        model_path = "classification/efficientnet_models//classify_5.pt"
        model_name = 'efficientnet-b7'
        class_names = {
            "000": "atopy",
            "001": "seborrheic dermatitis",
            "002": "psoriasis",
            "003": "rosacea",
            "004": "acne",

        }
    class Config_yolo():
        model_path = "segmentation/yolo_models/best_n.pt"
        model_names = None
        display = True
        save_path = None
        verbose = False
        device = 1
        label = True
        bbox = True
        segmentation = True
        file_paths = None


    ef_configs = [Config_41,Config_5]

    skin_lesion = Skin_lesion(ef_configs=ef_configs)
    image_path = "/home/dgdgksj/skin_lesion/ultralytics/atomom_test_images/"
    image_path_list = get_images_paths(image_path)
    output = []
    for i, image_path in enumerate(image_path_list):
        skin_lesion.inference(image_path=image_path)
        # output.append([file_name, class_name, confidence])
    # Write the output to a CSV file
    # with open('./experiment_results/output.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(output)
