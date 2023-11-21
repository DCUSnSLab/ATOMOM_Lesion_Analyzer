from similarity_matching import getChunk, compData_chunk, compData_full
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.db.models import Q
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
from rest_framework.parsers import JSONParser
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
curPath=os.getcwd()
path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
# print(path)
# print(os.listdir(path))
path=os.path.join(path,'atoOCR')

sys.path.append(path)
import demo_modifed_for_one_image_processing as ocr

os.chdir(path)
craftModel, model, opt = ocr.setModel()
os.chdir(curPath)
server_dir=os.getcwd()


cur1 = list(Product.objects.all().values_list('id', flat=True))
cur2=list(Product.objects.all().values_list('name', flat=True))
cur3 = list(Product.objects.all().values_list('brand', flat=True))

cur = zip(cur1,cur2,cur3)
origin_cur=zip(cur1,cur2,cur3)
#
# print(te[0])
cur = sorted(list(cur),key=lambda x : len(x[1]))
cur2 = sorted(list(Product.objects.all().values_list('name', flat=True)), key=len)

lenDict=getChunk(cur2)
def home(request):
    context = {}
    context['menutitle'] = 'HOME'

    return render(request, 'home.html', context)

def groupby(points,texts,img):
    rows,cols,_ =img.shape
    rThres = int(rows / 100)
    cThres = int(cols / 20)
    # newPoint

    datas=[]
    #포인트들에 텍스트 추가 (1,3,2,3,타이레놀)
    for i, point in enumerate(points):
        # print(i,point,texts[i])
        r1, c1, r2, c2 = point
        data=(r1,c1,r2,c2,texts[i])
        datas.append(data)

    check=False
    br=0
    splitIndex=[]
    for i, data in enumerate(datas):
        r1, c1,r2,c2,t = data
        if(check==False):
            br=r1
            check=True
            continue
        if(rThres<int(r1-br)):
            splitIndex.append(i)
        else:
            pass
        br=r1
    # for i in splitIndex:
    #     print(i)
    newTexts=""

    b=0
    newDatas=[]
    for i in range(len(splitIndex)):
        rowi=i
        i=splitIndex[i]
        data=datas[b:i]

        data = sorted(data, key=lambda x: (x[1]))
        b=i
        for i in data:
            newDatas.append(i)
        if(rowi==len(splitIndex)-1):
            # print(len(datas),i)
            data = datas[b:len(datas)]
            # print(data)
            # print(data)
            data = sorted(data, key=lambda x: (x[1]))
            for i in data:
                newDatas.append(i)
    # print("-"*50)
    # print(newDatas)
    newTexts=""
    check=False
    bdata=0
    # print(img.shape)
    centroid=0
    bcentroid=0
    rThres = int(rows / 30)
    for i, data in enumerate(newDatas):
        if(check==False):
            check=True
            bdata=data
            br1, bc1, br2, bc2, t = bdata
            newTexts+=t
            continue
        br1, bc1, br2, bc2, t = bdata
        r1, c1, r2, c2, t = data
        bcentroid=abs(br2-br1)+br1
        centroid=abs(r2-r1)+r1
        # print(bdata, data)
        # print("     bcentroid,centroid, rThres, int(abs(bcentroid-centroid))",bcentroid,centroid,rThres,int(abs(bcentroid-centroid)))
        if(rThres>int(abs(bcentroid-centroid))):

            # print("     ",cThres,abs(int(c1-bc2)))
            if(cThres>abs(int(c1-bc2))):
                newTexts+=' '
            else:
                newTexts += '\n'
        else:
            newTexts+='\n'
        newTexts+=t
        bdata=data
    # print(newTexts)
    return newTexts


def groupby_api_legacy(points,texts,rows,cols):
    rThres = int(rows / 100)
    cThres = int(cols / 20)
    # cThres = int(cols / 50)
    # newPoint

    datas=[]
    #포인트들에 텍스트 추가 (1,3,2,3,타이레놀)
    for i, point in enumerate(points):
        # print(i,point,texts[i])
        r1, c1, r2, c2 = point
        data=(r1,c1,r2,c2,texts[i])
        datas.append(data)



    check=False
    br=0
    splitIndex=[]
    for i, data in enumerate(datas):
        r1, c1,r2,c2,t = data
        if(check==False):
            br=r1
            check=True
            continue
        if(rThres<int(r1-br)):
            splitIndex.append(i)
            check=False
        else:
            pass
        br=r1
    # print("ddddd")
    # print(splitIndex)
    # print("ddddd")
    # for i in splitIndex:
    #     print(i)
    newTexts=""

    b=0
    newDatas=[]
    for i in range(len(splitIndex)):
        rowi=i
        i=splitIndex[i]
        data=datas[b:i]

        data = sorted(data, key=lambda x: (x[1]))
        b=i
        for i in data:
            newDatas.append(i)
        if(rowi==len(splitIndex)-1):
            # print(len(datas),i)
            data = datas[b:len(datas)]
            # print(data)
            # print(data)
            data = sorted(data, key=lambda x: (x[1]))
            for i in data:
                newDatas.append(i)
    # print("-"*50)
    # print(newDatas)
    newTexts=""
    check=False
    bdata=0
    # print(img.shape)
    centroid=0
    bcentroid=0
    rThres = int(rows / 30)
    for i, data in enumerate(newDatas):
        if(check==False):
            check=True
            bdata=data
            br1, bc1, br2, bc2, t = bdata
            newTexts+=t
            continue
        br1, bc1, br2, bc2, t = bdata
        r1, c1, r2, c2, t = data
        bcentroid=abs(br2-br1)+br1
        centroid=abs(r2-r1)+r1
        # print(bdata, data)
        # print("     bcentroid,centroid, rThres, int(abs(bcentroid-centroid))",bcentroid,centroid,rThres,int(abs(bcentroid-centroid)))
        if(rThres>int(abs(bcentroid-centroid))):

            # print("     ",cThres,abs(int(c1-bc2)))
            if(cThres>abs(int(c1-bc2))):
                newTexts+=' '
            else:
                newTexts += '\n'
        else:
            newTexts+='\n'
        newTexts+=t
        bdata=data
    # print(newTexts)
    return newTexts

def endLineCHeck(i,datas,newText,newTexts):
    if (i >= len(datas) - 1):
        newText = sorted(newText, key=lambda x: (abs(x[1])))
        newText = [i[4] for i in newText]
        # os.system('pause')
        newTexts += ' '.join(newText) + '\n'
        return newTexts
    else:
        return newTexts

def separate_list(datas):
    check = False
    newTexts = ""
    newText = []
    i = 0
    # print("ocr result")
    # print(datas)
    # print("*" * 50)
    for i, data in enumerate(datas):
        # print(data,i,len(datas)-1)
        if (check == False):
            # 라인 시작 부분 체크
            check = True
            bdata = data
            newText.append(data)
            continue
        br1, bc1, br2, bc2, t1 = bdata
        r1, c1, r2, c2, t2 = data
        range1 = range(br1, br2 + 1)
        range2 = range(r1, r2 + 1)
        x = set(range1)
        x = x.intersection(range2)
        if (len(x) == 0):
            # print(newText)
            newText = sorted(newText, key=lambda x: (abs(x[1])))
            newText=[i[4] for i in newText]
            # os.system('pause')
            newTexts += ' '.join(newText)+'\n'
            newText = []
            newText.append(data)
            bdata = data
            # print("교집합 원소수:",len(x), "기준 원소 수",len(range1) / 2,"     분리1")
            newTexts=endLineCHeck(i=i,datas=datas,newText=newText,newTexts=newTexts)
        elif ((len(range1) / 2.5 >= len(x))):
            newText = sorted(newText, key=lambda x: (abs(x[1])))
            # print(newText)
            newText = [i[4] for i in newText]
            newTexts += ' '.join(newText) + '\n'
            newText = []
            newText.append(data)
            bdata = data
            # print("교집합 원소수:",len(x), "기준 원소 수",len(range1) / 2,"     분리2")
            newTexts=endLineCHeck(i=i,datas=datas,newText=newText,newTexts=newTexts)
        else:
            newText.append(data)
            newTexts=endLineCHeck(i=i,datas=datas,newText=newText,newTexts=newTexts)
    # print("*"*50)
    # print(newTexts)
    return newTexts




def groupby_api(points,texts):


    #포인트들에 텍스트 추가 (1,3,2,3,타이레놀)
    datas=[(*points[i],texts[i]) for i in range(len(texts))]
    newTexts=separate_list(datas)
        # if( (r1>=br1 and r1<=br2) )

    return newTexts


def getMainProduct(curProduct):
    id = curProduct[1][0]
    brand = ""
    name = ""
    subName = ""
    barcode = ""
    pquery = Product.objects.raw(
        'select * from atomom_product where id =%s', [id])
    for q in pquery:
        id = q.id
        brand = q.brand
        name = q.name
        subName = q.subName
        barcode = q.barcode
    # print(name)
    iquery = Product.objects.raw(
        'select * from atomom_ingredients where id IN (select ingredients_id from atomom_pirelation where product_id=%s)',
        [id])
    ingredients = []
    for q in iquery:
        dic = dict(id=q.id, korean=q.korean, oldKorean=q.oldKorean, english=q.english,
                   oldEnglish=q.oldEnglish, hazardScoreMin=q.hazardScoreMin, hazardScoreMax=q.hazardScoreMax,
                   dataAvailability=q.dataAvailability, allergy=q.allergy, twenty=q.twenty, twentyDetail=q.twentyDetail,
                   goodForOily=q.goodForOily, goodForSensitive=q.goodForSensitive, goodForDry=q.goodForDry,
                   badForOily=q.badForOily, badForSensitive=q.badForSensitive, badForDry=q.badForDry,
                   skinRemarkG=q.skinRemarkG, skinRemarkB=q.skinRemarkB, cosmedical=q.cosmedical,
                   purpose=q.purpose, limitation=q.limitation, forbidden=q.forbidden
                   )
        # print(dic)
        # print(ingredients)
        ingredients.append(dic)

    main_data = {
        "notification":
            "1) 아토맘에서 제공하는 제품의 전성분은 브랜드사에서 수집한 정보입니다\n"
            "2) 구매 전에 제조판매업자가 표기한 전성분 표를 한 번 더 확인하시길 권장드립니다.\n"
            "3) 제품 뒷면에 표기되는 전성분과 아토맘에서 제공하는 전성분과 다를 시 피드백을 부탁드리겠습니다\n"
            "4) 아토맘 정보를 허가없이 수집 또는 활용할 경우, 법적 조치를 받을 수 있습니다.\n",
        "id": id,
        "brand": brand,
        "productName": name,
        "productSubName": subName,
        "barcode": barcode,
        "ingredients": ingredients
    }
    return (main_data,brand,name,id)

def getSubProducts(main_id,brand,main_name):
    sub_id=""
    subName = ""
    barcode = ""
    # main_id=39
    subProductDict=dict()
    pquery = Product.objects.raw(
        'select * from atomom_subproduct where product_id =%s', [main_id])
    cnt=0
    for q in pquery:
        sub_id=q.id
        subName = q.subName
        barcode=q.barcode
        iquery = Product.objects.raw(
            'select * from atomom_ingredients where id IN (select ingredients_id from atomom_spirelation where subproduct_id=%s)',
            [sub_id])
        ingredients = []
        for q in iquery:
            dic = dict(id=q.id, korean=q.korean, oldKorean=q.oldKorean, english=q.english,
                       oldEnglish=q.oldEnglish, hazardScoreMin=q.hazardScoreMin, hazardScoreMax=q.hazardScoreMax,
                       dataAvailability=q.dataAvailability, allergy=q.allergy, twenty=q.twenty,
                       twentyDetail=q.twentyDetail,
                       goodForOily=q.goodForOily, goodForSensitive=q.goodForSensitive, goodForDry=q.goodForDry,
                       badForOily=q.badForOily, badForSensitive=q.badForSensitive, badForDry=q.badForDry,
                       skinRemarkG=q.skinRemarkG, skinRemarkB=q.skinRemarkB, cosmedical=q.cosmedical,
                       purpose=q.purpose, limitation=q.limitation, forbidden=q.forbidden
                       )
            # print(dic)
            # print(ingredients)
            ingredients.append(dic)
        sub_data = {
            "notification":
                "1) 아토맘에서 제공하는 제품의 전성분은 브랜드사에서 수집한 정보입니다\n"
                "2) 구매 전에 제조판매업자가 표기한 전성분 표를 한 번 더 확인하시길 권장드립니다.\n"
                "3) 제품 뒷면에 표기되는 전성분과 아토맘에서 제공하는 전성분과 다를 시 피드백을 부탁드리겠습니다\n"
                "4) 아토맘 정보를 허가없이 수집 또는 활용할 경우, 법적 조치를 받을 수 있습니다.\n",
            "id": sub_id,
            "brand": brand,
            "productName": main_name,
            "productSubName": subName,
            "barcode": barcode,
            "ingredients": ingredients
        }
        # print(main_id,sub_id,subName,barcode)
        subProductDict[str(cnt)]=sub_data
        cnt+=1
    return (subProductDict,cnt)



def makePdata(curProduct):
    similarity = curProduct[2]
    main_data,brand,name,id =getMainProduct(curProduct)
    subProducts,cnt=getSubProducts(main_id=id,brand=brand,main_name=name)
    products=dict(mainProduct=main_data)
    products['nsubProduct']=cnt
    products["subProducts"]=subProducts
    products['similarity']=similarity
    pdata = {
        "products":  products
    }
    return pdata



# def get_line_result_chunk(lis,cur,lenDict,score1,score2,includeBrandKor=False,includeBrandEng=False):
#     nProduct=0
#     productList=[]
#     # print(' '.join(lis))
#     bestScore=0
#     best=(None,None)
#     for i, data in enumerate(lis):
#         check=False
#         # data="에코 에너지 위장 크림 [SPF50+/PA+++]"
#         # print(i, data)
#         if(includeBrandKor==True):
#             result1 = compData_chunk(cur, lenDict, data, score=score1, includeBrandKor=True)
#         elif(includeBrandEng==True):
#             result1 = compData_chunk(cur, lenDict, data, score=score1, includeBrandEng=True)
#         else:
#             result1= compData_chunk(cur, lenDict, data, score=score1)
#         curProduct = result1[len(result1) - 1]
#         if(curProduct[2]>=bestScore):
#             bestScore=curProduct[2]
#             pdata = makePdata(curProduct=curProduct)
#             nProduct = 1
#             productList=[pdata]
#             best=(nProduct,productList)
#
#
#     return best[0], best[1], bestScore

def get_line_result(lis,cur,score,includeBrandLeft=False,includeBrandRight=False):
    nProduct=0
    productList=[]
    # print(' '.join(lis))
    bestScore=0
    best=None
    for i, data in enumerate(lis):
        result1 = compData_full(cur, data, score=score,includeBrandLeft=includeBrandLeft,includeBrandRight=includeBrandRight)
        curResult, curScore = fullText_sub(data=result1)

        if (curResult == False and bestScore==0):
            best,bestScore=curResult,curScore
            pass
        else:
            check = getBest(curResult=curResult, curScore=curScore, best=best, bestScore=bestScore, score=score)
            if (check == True):
                return curResult,score
            else:
                best, bestScore = check

    # print("type:"+str(type(best)),"best:",str(best))
    return best, bestScore

def getFullText(cur,fullText,score):
    return compData_full(cur, fullText, score=score)
def getFullTextBrandLeft(cur,fullText,score):
    return compData_full(cur, fullText, score=score, includeBrandLeft=True)
def getFullTextBrandRight(cur,fullText,score):
    return compData_full(cur, fullText, score=score, includeBrandRight=True)
def fullText_sub(data):
    # print("d"+str(data)+"d")
    check=data[0]
    if(check[0]==None):
        return False,False
    else:
        bestScore=data[len(data)-1][2]
        nProduct=0
        productList=[]
        rev=data[::-1]
        for i, product in enumerate(rev):
            pData=makePdata(curProduct=product)
            nProduct+=1
            productList.append(pData)
        best=(nProduct, productList)
        return best,bestScore
def getBest(curResult,curScore,best,bestScore,score):
    if(curScore>=score):
        return True
    elif(curScore>=bestScore):
        best=curResult
        bestScore=curScore
        return best,bestScore
    else:
        return best,bestScore
def get_product(lis,cur,score):
    '''
    lis : line List ex) ['더페이스샾','스킨']
    cur : db product name, cur은 product name을 문자열 길이 수순으로 정렬되었습니다
    lenDict: {1:(0,23), 2:(23,230).... 문자열 길이 1은 cur[0:23]입니다
    score1 : compChunk를 이용해 targetText를 db와 비교하는데 이 비교를 중단하는 임계 깞입니다
    score2 : 줄단위 비교 또는 전체 비교를 수행하는데 이를 정답이라 인정 가능한 임계 값입니다
    '''
    bestScore=0
    best=None
    print("line")
    for i in lis:
        print(i)
    fullText = ' '.join(lis)
    print("fullText\n",fullText)
    print("len",len(list(deepcopy(cur))))
    experimentList=[]
    # print("len", len(list(cur)))
    t1=time.time()
    fullTextResult=getFullText(cur,fullText,score)
    curResult,curScore=fullText_sub(data=fullTextResult)
    t1=time.time()-t1
    print("\nfullText Result")
    if(curResult==False):
        print("     실패")
        experimentList.append((None, -1))
    else:
        check=getBest(curResult=curResult,curScore=curScore,best=best,bestScore=bestScore,score=score)
        if(check==True):
            return curResult
        else:
            print("     ", check[0][1][0]['products']['mainProduct']['brand'],
                  check[0][1][0]['products']['mainProduct']['productName'])
            best,bestScore=check
            experimentList.append((check[0][1][0]['products']['mainProduct']['brand'] +
                                   check[0][1][0]['products']['mainProduct']['productName'], t1))
    # print("len",len(list(cur)))
    # print("-"*50)
    t1 = time.time()
    fullTextLeft = getFullTextBrandLeft(cur,fullText,score)
    # print(fullTextLeft)
    curResult, curScore = fullText_sub(data=fullTextLeft)
    t1 = time.time() - t1
    print("fullText + bl")
    if (curResult == False):
        print("     실패")
        experimentList.append((None, -1))
    else:
        check = getBest(curResult=curResult, curScore=curScore, best=best, bestScore=bestScore, score=score)
        if (check == True):
            return curResult
        else:
            print("     ", check[0][1][0]['products']['mainProduct']['brand'],
                  check[0][1][0]['products']['mainProduct']['productName'])
            best, bestScore = check
            experimentList.append((check[0][1][0]['products']['mainProduct']['brand'] +
                                   check[0][1][0]['products']['mainProduct']['productName'], t1))
    t1 = time.time()
    fullTextRight = getFullTextBrandRight(cur,fullText,score)
    curResult, curScore = fullText_sub(data=fullTextRight)
    t1 = time.time() - t1
    print("fullText + br 실패")
    if (curResult == False):
        print("     실패")
        experimentList.append((None, -1))
    else:
        check = getBest(curResult=curResult, curScore=curScore, best=best, bestScore=bestScore, score=score)
        if (check == True):
            return curResult
        else:
            print("     ", check[0][1][0]['products']['mainProduct']['brand'],
                  check[0][1][0]['products']['mainProduct']['productName'])
            best, bestScore = check
            experimentList.append((check[0][1][0]['products']['mainProduct']['brand'] +
                                   check[0][1][0]['products']['mainProduct']['productName'], t1))
    t1 = time.time()
    curResult, curScore = get_line_result(lis=lis,cur=cur,score=score)
    t1 = time.time() - t1
    print("onlyLine")
    if (curResult == False):
        print("     실패")
        experimentList.append((None, -1))
    else:
        check = getBest(curResult=curResult, curScore=curScore, best=best, bestScore=bestScore, score=score)
        if (check == True):
            return curResult
        else:
            print("     ", check[0][1][0]['products']['mainProduct']['brand'],
                  check[0][1][0]['products']['mainProduct']['productName'])
            best, bestScore = check
            experimentList.append((check[0][1][0]['products']['mainProduct']['brand'] +
                                   check[0][1][0]['products']['mainProduct']['productName'], t1))
    t1 = time.time()
    curResult, curScore = get_line_result(lis=lis, cur=cur, score=score,includeBrandLeft=True)
    t1 = time.time() - t1
    print("line + bl 실패")
    if (curResult == False):
        print("     실패")
        experimentList.append((None, -1))
    else:
        check = getBest(curResult=curResult, curScore=curScore, best=best, bestScore=bestScore, score=score)
        if (check == True):
            return curResult
        else:
            print("     ", check[0][1][0]['products']['mainProduct']['brand'],
                  check[0][1][0]['products']['mainProduct']['productName'])
            best, bestScore = check
            experimentList.append((check[0][1][0]['products']['mainProduct']['brand'] +
                                   check[0][1][0]['products']['mainProduct']['productName'], t1))
    t1 = time.time()
    curResult, curScore = get_line_result(lis=lis, cur=cur, score=score,includeBrandRight=True)
    t1 = time.time() - t1
    print("line + br 실패")
    if (curResult == False):
        print("     실패")
        experimentList.append((None, -1))
    else:
        check = getBest(curResult=curResult, curScore=curScore, best=best, bestScore=bestScore, score=score)
        if (check == True):
            return curResult
        else:
            print("     ", check[0][1][0]['products']['mainProduct']['brand'],
                  check[0][1][0]['products']['mainProduct']['productName'])
            best, bestScore = check
            experimentList.append((check[0][1][0]['products']['mainProduct']['brand'] +
                                   check[0][1][0]['products']['mainProduct']['productName'], t1))

    if(bestScore==0):
        return None,experimentList
    else:
        return best,experimentList

@csrf_exempt
def api(request):
    os.chdir(server_dir)
    context = {}
    context['menutitle'] = 'OCR READ'
    # print("*"*50)
    print("\033[31mmethod", request.method)
    print("keys")
    print("     rows", request.GET.get('rows'))
    print("     cols", request.GET.get('cols'))
    print(request.GET)
    rows = request.GET.get('rows')
    cols = request.GET.get('cols')
    print("FILES'\033[0m'", request.FILES)
    print("cc",request.COOKIES)
    print(type(request))
    if 'media' in request.FILES:
        uploadfile = request.FILES.get('media', '')
        if uploadfile != '':
            print("여기 들어옴 ")
            # rows = int(request.COOKIES.get('rows', ''))
            # cols = int(request.COOKIES.get('cols', ''))
            # rows=int(rows)
            # cols=int(cols)
            name_old = uploadfile.name
            fs = FileSystemStorage(location='static/source')
            imgname = fs.save(f"src-{name_old}", uploadfile)

            imgPath=curPath+f"/static/source/{imgname}"
            os.chdir(path)
            t1=time.time()
            img, points = ocr.craftOperation(imgPath, craftModel, dirPath=opt.image_folder)

            texts = ocr.demo(opt,model)
            print("model time:",time.time()-t1)
            # for i, data in enumerate(points):
            #     print(data, texts[i])
            print(texts)
            # parsedText=groupby_api_legacy(points,texts,rows,cols)
            parsedText = groupby_api(points, texts)
            # print(parsedText)
            # parsedText = groupby_api(points, texts)
            # print("parsedText",parsedText)
            ocr.mkdir()
            os.chdir(curPath)
            lineList=parsedText.split('\n')
            # img = ocr.putText(img, points, texts)
            # print("*"*50)
            # print("texts",texts)
            # print(texts)
            # cv2.namedWindow("img",cv2.WINDOW_NORMAL)
            # cv2.imshow("img",img)
            # cv2.waitKey(0)

            q = Q()
            for i in texts:
                # print(i)
                q.add(Q(name__icontains=i), q.OR)
                q.add(Q(brand__icontains=i), q.OR)

            product = Product.objects.filter(q)
            product = list(product.values())
            if(len(product)==0):
                print("쿼리 결과 0")
                # product=origin_cur
                cur=list(deepcopy(origin_cur))


            else:
                cur = list((p['id'], p['name'], p['brand']) for p in product)
            best,lis = get_product(lineList, cur, score=150)
            for i in range(len(lis)):
                res, tim = lis[i]
                if (res == None or res == ""):
                    lis[i] = (None, -1)
            data = {
                "1_result": lis[0][0],
                "1_time": lis[0][1],
                "2_result": lis[1][0],
                "2_time": lis[1][1],
                "3_result": lis[2][0],
                "3_time": lis[2][1],
                "4_result": lis[3][0],
                "4_time": lis[3][1],
                "5_result": lis[4][0],
                "5_time": lis[4][1],
                "6_result": lis[5][0],
                "6_time": lis[5][1],
            }

            return JsonResponse(data)

    data = {
        "name": "파일을 읽을 수 없습니다 ",
        "status" : "cannot read file"
    }

    return JsonResponse(data)
@csrf_exempt
def coocr_upload(request):
    context = {}
    context['menutitle'] = 'OCR READ'

    imgname = ''
    resulttext = ''
    parsedText=''
    resultImgname=""
    print("*"*50)
    print("coocr")
    print("\033[31mmethod", request.method)
    print("FILES'\033[0m'", request.FILES)
    print()
    if request.method == 'POST' and not('uploadfile' in request.FILES):
        print(type(request))
        # file=request.FILES.get()
        # print(file)
        print("여기")

    if 'uploadfile' in request.FILES:
        uploadfile = request.FILES.get('uploadfile', '')

        if uploadfile != '':
            name_old = uploadfile.name
            name_ext = os.path.splitext(name_old)[1]

            fs = FileSystemStorage(location='static/source')
            imgname = fs.save(f"src-{name_old}", uploadfile)
            # print(uploadfile)
            # print(type(uploadfile))
            # print(imgname)


            imgfile = Image.open(f"./static/source/{imgname}")
            imgPath=curPath+f"./static/source/{imgname}"

            os.chdir(path)
            img, points = ocr.craftOperation(imgPath, craftModel, dirPath=opt.image_folder)

            texts = ocr.demo(opt,model)
            # print(points)
            # print(texts)
            parsedText=groupby(points,texts,img)

            # img = ocr.putText(img, points, texts)
            ocr.mkdir()

            resulttext = texts
            os.chdir(curPath)
            # print(f"{imgname}")
            # print(imgname)
            name, ext = os.path.splitext(imgname)
            # print('name :', name)
            # print('ext :', ext)
            resultImgname=name+"_result"+ext

            cv2.imwrite(curPath+f"./static/source/{resultImgname}",img)



    context['imgname'] = imgname
    context['resultImgname'] = resultImgname
    resulttext='\n'.join(resulttext)

    # context['resulttext'] = '\n'+resulttext
    context['resulttext'] = '\n' + parsedText
    # context['resulttext'] = parsedText

    # result1 = compData_chunk(cur, lenDict, target, score=90)
    lis=parsedText.split('\n')
    # print(lis)
    # print('*' * 50)

    productList=[]
    for i,data in enumerate(lis):
        # print(i,data)

        result1 = compData_chunk(cur, lenDict, data, score=95)
        curProduct=result1[len(result1)-1]
        if(curProduct[2]>=70):
            productList.append(curProduct)
            print("     ",curProduct)
            id = curProduct[1][0]
            query = Product.objects.raw(
                'select * from atomom_ingredients where id IN (select ingredients_id from atomom_pirelation where product_id=%s)',
                [id])
            for q in query:
                print("         ", q.korean)
    # for i in productList:
    #     print(i)
    #     id = i[0]
    #     query = Product.objects.raw(
    #         'select * from atomom_ingredients where id IN (select ingredients_id from atomom_pirelation where product_id=%s)',
    #         [id])
    #     for q in query:
    #         print("     ",q.korean)



    return render(request, 'coocr_upload.html', context)