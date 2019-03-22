class Evaluation:
    def __init__(self, list_ranked_docid, list_docid2relevance):
        self.list_ranked_docid = list_ranked_docid
        self.list_docid2relevance = list_docid2relevance
        self.list_sorted_docid = None
        
    def PrecisionK(self, threshold_k):
        
        precision_k = 0
        for index_of_relevance in self.list_ranked_docid[:threshold_k]:
            precision_k += self.list_docid2relevance[index_of_relevance]
            
        return precision_k
    
    def MAP(self):
        
        value_map = 0        
        for k in range(0,10):            
            value_map += self.PrecisionK(threshold_k=k)
        
        return value_map
        
        
    def NDCGp(self, p_value=10):
        
        temp_list_index = [i for i in range(len(self.list_docid2relevance))]
        self.list_sorted_docid = sorted(temp_list_index, key=lambda x:self.list_docid2relevance[x], reverse=True)
                
        value_idcgp = 0
        value_dcgp = 0
        for idx in range(0,p_value):
            i = idx 
            org_docid = self.list_ranked_docid[i]
            sorted_docid = self.list_sorted_docid[i]
            if i == 0:
                value_idcgp += self.list_docid2relevance[sorted_docid]
                value_dcgp += self.list_docid2relevance[org_docid]
            else:
                value_idcgp += self.list_docid2relevance[sorted_docid] / np.log2(i+1)
                value_dcgp += self.list_docid2relevance[org_docid] / np.log2(i+1)
        print(self.list_sorted_docid[:p_value])
        print(self.list_ranked_docid[:p_value])
        
        value_ndcgp = value_dcgp / value_idcgp
        return value_ndcgp        