import time
import json
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class cluteringDoc():
    '''
        extContent를 kmeans clustering하여 기사를 분류 함
    '''

    def __init__(self, num_cluster, sim_rate, docs):
        self.num_cluster = num_cluster
        self.sim_rate = sim_rate
        self.docs = docs
        self.fb_result = []

    def runClus(self):
        start = time.time()  # whole, entire

        # preprocessing
        df_data = pd.DataFrame([self.docs[x] for x in range(0, len(self.docs))])

        sentence = ""
        exceptSymbol = (
            'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'NF', 'NV', 'NA', 'ETM', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV',
            'JKQ',
            'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM')

        #issue 2019-10-30 pandas index error
        #df_data.reset_index(drop=True)

        for data, nid in zip(df_data['analyzed_text'], df_data['news_id']):
            for ssize in range(0, len(data['sentence'])):
                for data2 in data['sentence'][ssize]['WSD']:
                    if data2['type'] not in exceptSymbol:
                        sentence = sentence + data2['text'] + " "

            index = df_data.loc[df_data['news_id'] == nid].index.values[0]
            df_data.at[index, 'sentence_text'] = sentence
            sentence = ""
        #df_data = df_data.drop("analyzed_text", 1)

        # K-Means
        tvec = TfidfVectorizer()
        # pandas float error
        df_data['sentence_text'] = df_data['sentence_text'].astype(str)
        X = tvec.fit_transform(df_data['sentence_text'])

        model = KMeans(n_clusters=self.num_cluster, init='k-means++', max_iter=1000, n_init=100)
        tk_label = model.fit_predict(X)
        df_data["cluster_num"] = tk_label

        #No.1 get centroid value
        '''
        centro = model.cluster_centers_
        min=0
        cent=list()
        cent_num=list()
        new_X=X.toarray()
        for i in range(len(centro)):
            for j in range(len(new_X)):
                A = centro[i]
                B = new_X[j]
                dist=np.linalg.norm(A-B)
                if(min==0):
                    min = dist
                    sim_word=new_X[j]
                    sim_doc_count=j
                elif(min>dist):
                    min=dist
                    sim_word=new_X[j]
                    sim_doc_count = j
            min=0
            cent.append(sim_word)
            cent_num.append(sim_doc_count)
        print(cent_num)
        pre_cent_clus_doc=list()
        #centroid 문서 뽑기
        cnt=0
 
        for i in cent_num:
            for j, row in df_data.iterrows():
                if(i==j):
                    pre_cent_clus_doc.append(row['analyzed_text'])
                    break



        # 각 클러스터 별로 배열 만들기
        td = []
        each_cluster = []
        # initial
        for i in range(0, self.num_cluster):
            each_cluster.append([])
            td.append(None)


        for cnum, nid, content, analyzed_text in zip(df_data['cluster_num'], df_data['news_id'], df_data['extContent'], df_data['analyzed_text']):
            each_cluster[cnum].append((nid, content, analyzed_text))


        print("elapsed time taken for clustering: ", time.time() - start, "seconds.")
        elspedTime = time.time() - start

        #self.fb_result = pd_result

        return elspedTime, each_cluster, pre_cent_clus_doc
        '''
         #여기까지 centroid value

         #여기부터 whole value

        # 각 클러스터 별로 배열 만들기
        td = []
        each_cluster = []
        # initial
        for i in range(0, self.num_cluster):
            each_cluster.append([])
            td.append(None)

        '''for cnum, nid, content in zip(df_data['cluster_num'], df_data['news_id'], df_data['extContent']):
            each_cluster[cnum].append((nid, content))'''

        for cnum, nid, content, analyzed_text in zip(df_data['cluster_num'], df_data['news_id'], df_data['extContent'], df_data['analyzed_text']):
            each_cluster[cnum].append((nid, content, analyzed_text))


        print("elapsed time taken for clustering: ", time.time() - start, "seconds.")
        elapsedTime = time.time() - start

        #self.fb_result = pd_result

        return elapsedTime, each_cluster