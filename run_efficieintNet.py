import os
import numpy as np
from setproctitle import *
import csv
import skimage.draw
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import torch
import torch.onnx
import cv2
import natsort

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

class Efficient_net:
    def __init__(self, Config,device="cpu"):
        self.device = device
        self.Config = Config
        self.Config.num_classes = len(self.Config.class_names)
        self.model = self.__load_model()
        self.data_transforms1 = torch.nn.Sequential(
            transforms.Resize((224, 224)),
        )
        self.data_transforms2 = torch.nn.Sequential(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

    def __load_model(self):
        model = EfficientNet.from_pretrained(self.Config.model_name, num_classes=self.Config.num_classes)
        model.load_state_dict(torch.load(self.Config.model_path))
        model.eval()
        return model.to(self.device)

    def inference(self, image_info, verbose=False):
        image = None
        if isinstance(image_info, str):
            image = skimage.io.imread(image_info)
        elif isinstance(image_info, np.ndarray):
            #  cv2로 이미지를 읽는 상황이라 가정함
            image = cv2.cvtColor(image_info, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        assert image is not None, "경로 또는 ndarray를 입력하세요"


        T = transforms.ToTensor()
        inputs = self.data_transforms1(image)
        inputs = T(inputs).to(self.device)
        # inputs = inputs.to(self.device)
        inputs = self.data_transforms2(inputs)
        outputs = self.model(inputs.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # print(probabilities)
        top_p, top_class = probabilities.topk(self.Config.num_classes, dim=1)
        # print(top_p, top_class )
        result = list(
            zip([t.item() for t in top_p.squeeze().squeeze()], [t.item() for t in top_class.squeeze().squeeze()]))
        result.sort(key=lambda x: x[1])
        # print(result)
        classification_result = [None for _ in range(self.Config.num_classes)]
        for i, data in enumerate(result):
            class_prob, class_id = data
            class_name = self.Config.class_names[str(class_id).zfill(3)]
            classification_result[i] = (class_name, class_prob)
        # print(classification_result)
        classification_result.sort(key=lambda x: x[1], reverse=True)
        if (self.Config.verbose):
            print(classification_result, "?")
        # print("self.Config.topk",self.Config.topk)
        return classification_result[:self.Config.topk]



class Skin_lesion:
    def __init__(self,Config_41=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cwd = os.getcwd()
        self.Config_41 = Config_41
        self.ef_41 = self.__load_ef_net(Config=self.Config_41,device=self.device)

        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __load_ef_net(self,Config,device):
        # os.chdir('./classification')
        ef_net = Efficient_net(Config=Config,device=device)
        # os.chdir(self.cwd)
        return ef_net

    def inference(self, image_path):
        class_name, confidence = self.ef_41.inference(image_path=image_path)[0]
        file_name = os.path.basename(image_path).split('.')[0]
        print(file_name,class_name,confidence)
        return file_name,class_name,confidence




if __name__ == '__main__':
    setproctitle('lesion')

    class Config_41():
        verbose = False
        topk = 3
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


    image_path = "/home/dgdgksj/skin_lesion/ultralytics/atomom_test_images/"
    skin_lesion = Skin_lesion(Config_41=Config_41)

    image_path_list = get_images_paths(image_path)
    output = []
    for i, image_path in enumerate(image_path_list):
        file_name,class_name,confidence = skin_lesion.inference(image_path=image_path)
        output.append([file_name, class_name, confidence])
    # Write the output to a CSV file
    with open('./experiment_results/output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(output)
