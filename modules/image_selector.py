from modules.base_module import BaseModule
from modules.image_selection.image_selection_2019_v1 import *

class ImageSelectionModule(BaseModule):
    def __init__(self, topic, out_path):        
        self.topic = topic
        self.out_path = out_path

    def process_data(self, query, image_save_path, download_limit = 50):
        if type(query) is not list:
            topic = []
            topic.append(query)
        else:
            topic = query
        file_list, image_list, caption_list = image_caption_downloader(topic, download_limit) # get candidate images and caption from google
        nongraph_image_list, nongraph_caption_list = VGG_classifier(file_list, image_list, caption_list) # filter images using vgg classifier 
        final_image = semantic_similarity_module(topic, nongraph_image_list, nongraph_caption_list) # get final image through semantic similarity measurement
        imgsave_path = image_save_path # path to save final recommended image
        temp_image = Image.open(final_image)
        temp_image.save('%s%s'%(imgsave_path,final_image.split('/')[3])) # save final recommended image to given path
        shutil.rmtree('./downloads/%s'%(final_image.split('/')[2]), ignore_errors=True)
