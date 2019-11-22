from modules.base_module import BaseModule
from modules.image_selection.image_selection_2019_v2 import *
from keras.models import load_model

class ImageSelectionModule(BaseModule):
    def __init__(self, out_path):        
        self.out_path = out_path
        self.vggModel = load_model('image_selection/weight_best.hdf') ###
        self.g = tf.Graph()
        with self.g.as_default():
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
            self.embedded_text = self.embed(self.text_input)
            self.init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        self.g.finalize()
    def process_data(self, query, image_save_path, download_limit = 50):
        if type(query) is list:
            topic = []
            topic.append(query[0].replace('/','_'))
        if type(query) is not list:
            query = query.replace('/','_')
            topic = []
            topic.append(query)
        while True: # handling VGG16 input error. because of image download (such as page not found)
            try:
                file_list, image_list, caption_list = image_caption_downloader(topic, download_limit)
                temp_f = file_list[0] 
                break
            except IndexError:
        print("image and caption downloaded")
        nongraph_image_list, nongraph_caption_list = VGG_classifier(file_list, image_list, caption_list, self.vggModel) ###
        final_image = semantic_similarity_module(self.g, self.init_op, self.embedded_text, self.text_input, topic, nongraph_image_list, nongraph_caption_list) ####
        imgsave_path = image_save_path # path to save final recommended image
        temp_image = Image.open(final_image)
        temp_image = temp_image.convert('RGB')
        temp_image.save('%s%s'%(imgsave_path,final_image.split('/')[3]))
        shutil.rmtree('./downloads/%s'%(final_image.split('/')[2]), ignore_errors=True)


