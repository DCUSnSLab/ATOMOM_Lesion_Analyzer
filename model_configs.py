import os
from mrcnn.config import Config as Mask_RCNN_Config

class Config_eff():
	weight = None
	device = None
	verbose = None
	topk = None
	model_path = None
	model_name = None
	class_names = None

class Cfg_1st_EffB7_Su_Cls_5(Config_eff):
	# note Config_first_learning_result_efficientNet_b7_su_class_5
	weight = 0.1
	device = 1
	verbose = False
	topk = 5
	model_path = "classification/efficientnet_models/first_learning_result_efficientNet_b7_su_class_5.pt"
	model_name = 'efficientnet-b7'
	class_names = {
		"000": "atopy",
		"001": "seborrheic dermatitis",
		"002": "psoriasis",
		"003": "rosacea",
		"004": "acne",
	}

class Cfg_2nd_EffB0_Su_Cls_41(Config_eff):
	# note Config_second_learning_result_efficientNet_b0_su_class_41
	weight = 0.5
	device = 0
	verbose = False
	topk = 5
	model_path = os.path.join(os.getcwd(), "classification/efficientnet_models/second_learning_result_efficientNet_b0_su_class_41.pt")
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

class Cfg_2nd_EffB0_Ming_Cls_41(Config_eff):
	# note Config_second_learning_result_efficientNet_b0_mingeon_class_41
	weight = 0.1
	device = 0
	verbose = False
	topk = 5
	model_path = os.path.join(os.getcwd(), "classification/efficientnet_models/second_learning_result_efficientNet_b0_mingeon_class_41.pt")
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

class Cfg_2nd_EffB0_Ming_Cls_6(Config_eff):
	# note Config_second_learning_result_efficientNet_b0_mingeon_class_6
	weight = 0.1
	device = 1
	verbose = False
	topk = 5
	model_path = "classification/efficientnet_models/second_learning_result_efficientNet_b0_mingeon_class_6.pt"
	model_name = 'efficientnet-b0'
	class_names = {
		"000": "normal_skin",
		"001": "atopy",
		"002": "psoriasis",
		"003": "acne",
		"004": "epidermal_cyst",
		"005": "varicella",
	}

class Cfg_2nd_EffB7_Ming_Cls_6(Config_eff):
	# note Config_second_learning_result_efficientNet_b7_mingeon_class_6
	weight = 0.1
	device = 1
	verbose = False
	topk = 5
	model_path = "classification/efficientnet_models/second_learning_result_efficientNet_b7_mingeon_class_6.pt"
	model_name = 'efficientnet-b7'
	class_names = {
		"000": "normal_skin",
		"001": "atopy",
		"002": "psoriasis",
		"003": "acne",
		"004": "epidermal_cyst",
		"005": "varicella",
	}

class Cfg_3rd_EffB0_Ming1_Cls_4(Config_eff):
	# note Config_third_learning_result_efficientNet_b0_mingeon_1_class_4
	weight = 0.1
	device = 1
	verbose = False
	topk = 5
	model_path = "classification/efficientnet_models/third_learning_result_efficientNet_b0_mingeon_1_class_4.pt"
	model_name = 'efficientnet-b0'
	class_names = {
		"000": "normal_skin",
		"001": "atopy",
		"002": "psoriasis",
		"003": "urticaria",
	}
class Cfg_3rd_EffB0_Ming2_Cls_4(Config_eff):
	# note Config_third_learning_result_efficientNet_b0_mingeon_2_class_4
	weight = 0.1
	device = 1
	verbose = False
	topk = 5
	model_path = "classification/efficientnet_models/third_learning_result_efficientNet_b0_mingeon_2_class_4.pt"
	model_name = 'efficientnet-b0'
	class_names = {
		"000": "normal_skin",
		"001": "atopy",
		"002": "psoriasis",
		"003": "urticaria",
	}
class Cfg_3rd_EffB0_Ming3_Cls_4(Config_eff):
	# note Config_third_learning_result_efficientNet_b0_mingeon_3_class_4
	weight = 0.5
	device = 0
	verbose = False
	topk = 5
	model_path = "classification/efficientnet_models/third_learning_result_efficientNet_b0_mingeon_3_class_4.pt"
	model_name = 'efficientnet-b0'
	class_names = {
		"000": "normal_skin",
		"001": "atopy",
		"002": "psoriasis",
		"003": "urticaria",
	}
class Cfg_3rd_EffB0_Ming4_Cls_4(Config_eff):
	# note Config_third_learning_result_efficientNet_b0_mingeon_4_class_4
	weight = 0.1
	device = 1
	verbose = False
	topk = 5
	model_path = "classification/efficientnet_models/third_learning_result_efficientNet_b0_mingeon_4_class_4.pt"
	model_name = 'efficientnet-b0'
	class_names = {
		"000": "normal_skin",
		"001": "atopy",
		"002": "psoriasis",
		"003": "urticaria",
	}

class Config_yolo():
	model_path = "segmentation/yolo_models/best_n.pt"
	model_names = None
	display = False
	save_path = None
	verbose = False
	device = 0
	label = False
	bbox = False
	segmentation = True
	file_paths = None


class Config_mrcnn(Mask_RCNN_Config):
	ROOT_DIR = os.path.abspath("./")
	COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
	DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
	# fixme 클래스네임이 정확하지 않아요 기입을 안하면 터지네요
	class_names = ['others', 'atopic_dermatitis', "seborrheic dermatitis", "psoriasis", "rosacea", "acne"]
	weights = './segmentation/mrcnn_models/mask_rcnn_atopy_0035.h5'
	NAME = "atopy"
	logs = DEFAULT_LOGS_DIR,
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 1
	VALIDATION_STEPS = 50
	STEPS_PER_EPOCH = 1000
	DETECTION_MIN_CONFIDENCE = 0.9
# GPU_COUNT = 1
# IMAGES_PER_GPU = 1