# -*- coding: utf-8 -*- 
import urllib.request, urllib.parse, urllib.error
import os
import json
from bs4 import BeautifulSoup

# packages for universal sentence encoder - multilingual
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

# packages for vgg classifier
import os, re, glob
import numpy as np
import shutil
from numpy import argmax
from keras.models import load_model
from PIL import Image
import PIL.Image as pilimg

from scipy import spatial


#---------image and title download--------------
def image_caption_downloader(query, download_limit):
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
    for l in link:
        filename = "%d.jpg" % (i)
        file_list.append(filename)
        fullfilename = os.path.join(foldername, filename)
        (filename, h) = urllib.request.urlretrieve(l[1], fullfilename)
        caption_list.append("%s"%(l[2]))
        image_list.append(fullfilename)
        i = i + 1
        if i > download_limit:
            break
    k = k + 1
    return file_list, image_list, caption_list


#-----------Universal Sentence Encoder - Multilingual Embedding------------------
def USE_embedding(input_text):
    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
        embedded_text = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()
    session = tf.Session(graph=g)
    session.run(init_op)
    embedded_output = session.run(embedded_text, feed_dict={text_input: input_text})
    return embedded_output


#-----------VGG graph/non-graph image classifier--------------------------------
def VGG_classifier(file_list, image_list, caption_list):
    resized_image_list = []
    for i in image_list:
        resized = Image.open(i).convert('RGB').resize((224,224))
        pix=np.array(resized) / 255
        resized_image_list.append(pix)
    categories = ["graph","others"] # label 0 : graph image / label 1 : non-graph
    test = np.array(resized_image_list)
    model = load_model('weight_best.hdf')
    predict = model.predict_classes(test)
    for i in range(len(test)):
        print(file_list[i] + " : , Predict : "+ str(categories[predict[i]]))
    nongraph_image_list = []
    nongraph_caption_list = []
    for i in range(len(test)):
        if predict[i] == 1:
            nongraph_image_list.append(image_list[i])
            nongraph_caption_list.append(caption_list[i])
    return nongraph_image_list, nongraph_caption_list


#-----------calculate semantic similarity and recommend image-----------------------
def semantic_similarity_module(query, nongraph_image_list, nongraph_caption_list):
    query_embedding = USE_embedding(query)
    nongraph_caption_embedding = USE_embedding(nongraph_caption_list)
    DC = [[0 for x in range(len(nongraph_caption_list))] for x in range(len(query))]
    for i in range(len(query)):
        for j in range(len(nongraph_caption_list)):
            DC[i][j] = spatial.distance.cosine(query_embedding, nongraph_caption_embedding[j])
    final_image = nongraph_image_list[DC[0].index(min(DC[0]))]
    print('recommended image : %s'%(final_image.split('/')[3]))
    return final_image


#-----------------------------------------------------------------
query = ['2000억 달러 상당 중국산 제품 관세율 인상']
download_limit = 50 # the number of candidate image/caption pairs to download
file_list, image_list, caption_list = image_caption_downloader(query, download_limit)
nongraph_image_list, nongraph_caption_list = VGG_classifier(file_list, image_list, caption_list)
final_image = semantic_similarity_module(query, nongraph_image_list, nongraph_caption_list)



