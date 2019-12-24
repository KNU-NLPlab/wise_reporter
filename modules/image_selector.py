from modules.base_module import BaseModule
from modules.image_selection.image_selection_2019_v2 import *
from keras.models import load_model
from elasticsearch import Elasticsearch

class ImageSelectionModule(BaseModule):
    def __init__(self, out_path):        
        self.out_path = out_path
        self.vggModel = load_model('modules/image_selection/weight_best.hdf') ###
        self.es = Elasticsearch("155.230.34.145:9200")
        print("weight_best loaded")
        self.g = tf.Graph()
        with self.g.as_default():
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
            self.embedded_text = self.embed(self.text_input)
            self.init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        self.g.finalize()
        print("graph defined")
        
    def process_data(self, query, image_save_path, download_limit = 50, silence=True):
        es_query = {'query':{'match':{'keyword':query}}}
        res = self.es.search(index='image', body=es_query)
        if res['hits']['total']['value'] == 0:
            output = None
        else:
            output = res['hits']['hits'][0]['_source']
        path_cached = ('./downloads')
        check_query_exist = [path[:50] for path in os.listdir(path_cached)]
        if query not in check_query_exist:  # case 1 : if query is new - download images & caption from google, scouter caching(url, caption), local caching(image path, caption)
            if not silence:
                print("query is new")
            if type(query) is list:
                topic = []
                topic.append(query[0].replace('/','_'))
            if type(query) is not list:
                query = query.replace('/','_')
                topic = []
                topic.append(query)
            if not silence:
                print("query pre-processed")
            try_iter = 1
            while try_iter < 5: # handling VGG16 input error, no image found in google
                try:
                    dict_image, file_list, image_list, caption_list = image_caption_downloader(topic, download_limit, silence)
                    temp_f = file_list[0] 
                    break
                except IndexError:
                    print("try image download again")
                    try_iter = try_iter + 1

            if len(file_list) == 0:
                shutil.rmtree('./downloads/%s'%(query), ignore_errors=True)
                print("query does 222")
                return False

            # --- scouter caching ---
            if output == None:
                image_dict = { 'keyword': query, 'image': dict_image[query][0], 'caption': dict_image[query][1]} 
                res = self.es.index(index='image', body=image_dict)
            else:
                if not silence:
                    print("query already cached in scouter")

            # --- local caching ---
            try:
                if not(os.path.isdir('localcache')):
                    os.makedirs(os.path.join('localcache'))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    print("Failed to create directory!!!!!")
                    raise
            localcache_img_name = 'localcache/%s_img.txt'%(query[:50])
            localcache_caption_name = 'localcache/%s_caption.txt'%(query[:50])
            with open(localcache_img_name, 'w') as f:
                for imgfile in image_list:
                    f.write("%s\n"%imgfile)
            f.close()
            with open(localcache_caption_name, 'w')as f2:
                for captionfile in caption_list:
                    f2.write("%s\n"%captionfile)
            f2.close()

        else: # case 2 : if query is used before - use local cached data
            try:
                if not silence:
                    print("query already used before")
                localcache_img_name = 'localcache/%s_img.txt'%(query[:50])
                localcache_caption_name = 'localcache/%s_caption.txt'%(query[:50])
                file_list, image_list, caption_list = image_caption_get(localcache_img_name, localcache_caption_name, silence)
            except IOError:
                print("query does not have cached images and captions")
                return False

        nongraph_image_list, nongraph_caption_list = VGG_classifier(file_list, image_list, caption_list, self.vggModel, silence) ###
        final_image = semantic_similarity_module(self.g, self.init_op, self.embedded_text, self.text_input,
                                                 topic, nongraph_image_list, nongraph_caption_list, silence) ####
        if final_image is None:
            return False
        imgsave_path = image_save_path # path to save final recommended image
        temp_image = Image.open(final_image)
        temp_image = temp_image.convert('RGB')
        temp_image.save('%s'%(imgsave_path))
        
        return True
        #shutil.rmtree('./downloads/%s'%(final_image.split('/')[2]), ignore_errors=True)

