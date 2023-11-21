
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.db.models import Q

from datetime import datetime
import os, sys
from PIL import Image
import cv2
import time
from .models import Product
import numpy as np
from copy import deepcopy
import django
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from silk.profiling.profiler import silk_profile
import csv
from rest_framework.parsers import JSONParser
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
# django.setup()
# curPath=os.getcwd()
# path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
# # print(path)
# # print(os.listdir(path))
# path=os.path.join(path,'atoOCR')
#
# sys.path.append(path)
#
#
#
# os.chdir(path)
# craftModel, model, opt = ocr.setModel()
# os.chdir(curPath)
# server_dir=os.getcwd()

curPath=os.getcwd()


def load_models(path):
    temp_path = os.getcwd()
    print("sd",path)
    os.chdir(path)
    os.chdir('../')
    sys.path.append(os.getcwd())
    sys.path.append(path)
    sys.path.append(os.path.join(path, 'segmentation'))
    sys.path.append(os.path.join(path, 'classifiaction'))
    print("sd2",path)
    import run_demo
    import model_configs
    # os.chdir(temp_path)
    # print("dsd",os.getcwd())
    # print(path)
    # print(os.path.join(path, 'segmentation'))
    # print(os.path.join(path, 'classifiaction'))

    # # note 0.89 acc 0.45 0.35
    # ef_configs = [model_configs.Cfg_2nd_EffB0_Su_Cls_41, model_configs.Cfg_2nd_EffB0_Ming_Cls_6]
    # ef_configs = [model_configs.Cfg_1st_EffB7_Su_Cls_5,model_configs.Cfg_2nd_EffB0_Su_Cls_41,model_configs.Cfg_2nd_EffB0_Ming_Cls_41,model_configs.Cfg_2nd_EffB0_Ming_Cls_6,model_configs.Cfg_2nd_EffB7_Ming_Cls_6,
    # #               model_configs.Cfg_3rd_EffB0_Ming1_Cls_4,model_configs.Cfg_3rd_EffB0_Ming2_Cls_4,model_configs.Cfg_3rd_EffB0_Ming3_Cls_4,model_configs.Cfg_3rd_EffB0_Ming4_Cls_4]
    ef_configs = [model_configs.Cfg_2nd_EffB0_Su_Cls_41, model_configs.Cfg_3rd_EffB0_Ming3_Cls_4]
    # # ef_configs = [Config_6_min]
    yolo_configs = [model_configs.Config_yolo]
    mrcnn_configs = []
    skin_lesion = run_demo.Skin_lesion(ef_configs=ef_configs, yolo_configs=yolo_configs, mrcnn_configs=mrcnn_configs)
    return skin_lesion
    # image_path = "test_data/atomom_test_images_samples/"
    #
    # image_path_list = get_images_paths(image_path)
    # output = []
    # for i in tqdm(range(len(image_path_list))):
    #     image_path = image_path_list[i]
    #     # print(image_path)
    #     skin_lesion.inference(image_path=image_path)
    #     # break
    #
    # if (skin_lesion.exp):
    #     skin_lesion.save_exp()

print("1")
skin_lesion = load_models(curPath)
os.chdir(curPath)
print("2")

@silk_profile(name='Lesion_Detect_Profile')
@csrf_exempt
def home(request):
    global curPath
    context = {}
    context['menutitle'] = '아토'
    context['datas'] = [1,2,3,4,5,6,7,8]
    context['originaltext'] = ''
    context['resulttext'] = ''
    error_message = {
        "name": "이미지 파일을 읽을 수 없습니다 ",
        "status": "cannot read file"
    }

    if request.method == 'POST' and ('uploadfile' in request.FILES):
        print(type(request))
        # file=request.FILES.get()
        # print(file)
        print("여기")

    if 'uploadfile' in request.FILES:
        uploadfile = request.FILES.get('uploadfile', '')

        if uploadfile != '':
            name_old = uploadfile.name
            name_ext = os.path.splitext(name_old)[1]
            print("여기2")
            fs = FileSystemStorage(location='static/source')
            imgname = fs.save(f"src-{name_old}", uploadfile)
            img_path = fs.path(f"src-{name_old}")

            # print(f"src-{name_old}", uploadfile)
            # img = cv2.imread(img_path)
            name, ext = os.path.splitext(imgname)




            srcImgname = name + ext
            resultImgname = name + "result.jpg"
            yolo_inferred_images, soft_voting_result = skin_lesion.inference(image_path=img_path)
            cv2.imwrite(os.path.dirname(img_path) + "/" + name + "result.jpg", yolo_inferred_images)
            # print(soft_voting_result)
            context['resultImgname'] = resultImgname
            context['srcImgname'] = srcImgname

            formatted_data_ori = ""
            formatted_data_res = ""

            for i in range(0, len(soft_voting_result), 2):
                probability = round(soft_voting_result[i + 1] * 100, 4)
                formatted_data_ori += f'{soft_voting_result[i]}: {soft_voting_result[i + 1]}\n'
                if probability >= 1:
                    formatted_data_res += f'{soft_voting_result[i]}: {probability}%\n'
                # else:
                #     formatted_data_res += f'{soft_voting_result[i]}\n'
            # print([formatted_data])
            formatted_data_ori = formatted_data_ori.strip()
            formatted_data_res = formatted_data_res.strip()

            context['originaltext'] = formatted_data_ori
            context['resulttext'] = formatted_data_res
    print([context['originaltext']])
    # print(resultImgname)
    # if 'media' not in request.FILES or request.FILES.get('media', '') == '':
    #     return JsonResponse(error_message)
    # os.chdir(server_dir)
    # uploadfile = request.FILES.get('media', '')
    # name_old = uploadfile.name
    # fs = FileSystemStorage(location='static/source')
    # imgname = fs.save(f"src-{name_old}", uploadfile)
    # imgPath = curPath + f"/static/source/{imgname}"

    # context['resulttext'] = parsedText




    return render(request, 'home.html', context)