# -*- coding: utf-8 -*- 
import urllib.request, urllib.parse, urllib.error
import os
import json
from bs4 import BeautifulSoup
import tensorflow as tf 
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece
import os, re, glob
import shutil
from numpy import argmax
from keras.models import load_model
from PIL import Image
from scipy import spatial

os.environ['CUDA_VISIBLE_DEVICES']='0'

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 #########
set_session = tf.Session(config=config)



#---------sentence embedding module--------------
def USE_embedding(input_text, g, init_op, embedded_text, text_input):
    session = tf.Session(graph=g)
    session.run(init_op)
    embedded_output = session.run(embedded_text, feed_dict={text_input: input_text})
    return embedded_output


#---------get image and caption from local cached data--------------
def image_caption_get(localcache_img_name, localcache_caption_name, silence):
    file_list = []
    image_list = []
    caption_list = []
    f = open(localcache_img_name,'r')
    while True:
        line = f.readline()
        if not line: break
        file_list.append((line.split('/')[-1]).splitlines()[0])
        image_list.append(line.splitlines()[0])
    f.close()
    if not silence:
        print(file_list)
        print(image_list)
    f2 = open(localcache_caption_name,'r')
    while True:
        line = f2.readline()
        if not line: break
        caption_list.append(line.splitlines()[0])
    f2.close()
    if not silence:
        print(caption_list)
    return file_list, image_list, caption_list


#---------get image and caption from google search engine--------------
def image_caption_downloader(query, download_limit, silence):
    try:
        if not(os.path.isdir('downloads')):
            os.makedirs(os.path.join('downloads'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
    file_list= [] # only file name
    image_list = [] # path + file name 
    caption_list = []
    header={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    k = 1
    i = 1
    url = "https://www.google.co.kr/search?q=" + urllib.parse.quote(query[0]) + "&source=lnms&tbm=isch" 
    foldername = "./downloads" + "/" + query[0] 
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    req = urllib.request.Request(url, headers=header)
    response = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(response, "html.parser")
    link = [(json.loads(div.text)['ou'], json.loads(div.text)['tu'], json.loads(div.text)['pt']) for div in soup('div', 'rg_meta')]
    dict_image = {} # dictionary for caching images to scouter
    image_url_list = []
    for l in link:
        filename = "%d.jpg" % (i)
        file_list.append(filename)
        fullfilename = os.path.join(foldername, filename)
        image_url_list.append(l[1])
        (filename, h) = urllib.request.urlretrieve(l[1], fullfilename)
        caption_list.append("%s"%(l[2]))
        image_list.append(fullfilename)
        i = i + 1
        if i > download_limit:
            break
    k = k + 1
    dict_image[query[0]] = (image_url_list, caption_list)
    return dict_image, file_list, image_list, caption_list


#-----------VGG graph/non-graph image classifier--------------------------------
def VGG_classifier(file_list, image_list, caption_list, vggModel, silence):
    resized_image_list = []  
    for i in range(len(image_list)):
        resized = Image.open(image_list[i]).convert('RGB').resize((224,224))
        pix=np.array(resized) / 255
        resized_image_list.append(pix)
    categories = ["graph","others"] # label 0 : graph image / label 1 : non-graph
    test = np.array(resized_image_list)
    if not silence:
        print(test.shape)
    predict = vggModel.predict_classes(test)
    if not silence:
        print("vgg predict - success")
    for i in range(len(test)):
        if not silence:
            print(file_list[i] + " : , Predict : "+ str(categories[predict[i]]))
    nongraph_image_list = []
    nongraph_caption_list = []
    for i in range(len(test)):
        if predict[i] == 1:
            nongraph_image_list.append(image_list[i])
            nongraph_caption_list.append(caption_list[i])
    return nongraph_image_list, nongraph_caption_list


#-----------calculate semantic similarity and recommend image-----------------------
def semantic_similarity_module(g, init_op, embedded_text, text_input, query, nongraph_image_list, nongraph_caption_list, silence):
    try:
        query_embedding = USE_embedding(query, g, init_op, embedded_text, text_input)
        nongraph_caption_embedding = USE_embedding(nongraph_caption_list,  g, init_op, embedded_text, text_input)
        DC = [[0 for x in range(len(nongraph_caption_list))] for x in range(len(query))]
        for i in range(len(query)):
            for j in range(len(nongraph_caption_list)):
                DC[i][j] = spatial.distance.cosine(query_embedding, nongraph_caption_embedding[j])
        final_image = nongraph_image_list[DC[0].index(min(DC[0]))]
    except:
        return None
    return final_image
