from modules.base_module import BaseModule

from modules.scouter import connectES as es
from modules.scouter import clusteringDoc as cl
from modules.scouter import N_gram as ng
import json


class TopicGenerator(BaseModule):
    
    def __init__(self, topic, out_path):
        super(TopicGenerator, self).__init__(topic, out_path)
        self.getDoc = es.ScouterHandler()
                 
    def process_data(self, date):
        indexed_cluster = es.keywordGetter(date)
        
        if indexed_cluster != False:
            return indexed_cluster
        else:
            qbody = es.ScouterHandler.make_date_query_body(date,("news_id","extContent","analyzed_text","analyzed_text.sentence.WSD.text",
                                                                 "analyzed_text.sentence.WSD.type","category","analyzed_title.sentence.WSD.text",
                                                                 "analyzed_title.sentence.WSD.type"))
          
            docs  = self.getDoc.search(qbody,'newspaper')
            result = {'date': date,
               'doc count': len(docs)}
            clus = cl.cluteringDoc(5,0.95,docs)

           #centroivalue 용
           #ctime, c_result, c_cent_result = clus.runClus()
           #for i in c_cent_result:
           #    c_doc.append([i])

           #whole value 용
            ctime, c_result = clus.runClus()
            c_doc=list()
            for i in range(len(c_result)):
                analyzed_doc = list()
                for j in c_result[i]:
                    analyzed_doc.append(j[2])
                c_doc.append(analyzed_doc)

            '''analyzed_doc = list()
               for j in c_cent_result:
                   analyzed_doc.append(j)
               c_doc.append(analyzed_doc)'''

           #print(1)
            for i in range(5):
                cluster_uni, cluster_bi, cluster_tri=ng.ngram(c_doc[i])
                cluster_uni = cluster_uni.most_common(10)
                cluster_bi = cluster_bi.most_common(10)
                cluster_tri = cluster_tri.most_common(10)
               # a=sorted(cluster_uni.items(), key=lambda t:t[1], reverse=True)
               # b=sorted(cluster_bi.items(), key=lambda t:t[1], reverse=True)
               # c=sorted(cluster_tri.items(), key=lambda t:t[1], reverse=True)
                result['cluster'+str(i+1)+'unigram'] = cluster_uni
                result['cluster'+str(i+1)+'bigram'] = cluster_bi
                result['cluster'+str(i+1)+'trigram'] = cluster_tri
                
            return result
    
    
    



