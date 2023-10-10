import os,shutil,glob
from pathlib import Path
from icrawler.builtin import BingImageCrawler
from PIL import Image

efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
# Setup data inputs
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32

def resize_img(img):
    image = Image.open(img)
    new_image = image.resize(IMAGE_SHAPE)
    new_image.save(img)

def download_image_from_bing(word, n_image):
    Bing_Crawler = BingImageCrawler(storage = {'root_dir': str(Path.cwd())+'\\tmp_'+word})
    Bing_Crawler.crawl(keyword = word, max_num = n_image)
    for img in glob.glob(str(Path.cwd())+'\\tmp_'+word+'\\*'):
        resize_img(img)

def partition_list(list, percentage):
    idx = int(len(list) * percentage)
    return list[:idx], list[idx:]    

def make_train_and_test(source, percentage):
    dataset = [x for x in glob.glob(str(Path.cwd())+'\\tmp_'+source+'\\*')]
    train, test = partition_list(dataset ,percentage)
    for (file_list, dirname) in ((train, str(Path.cwd())+'\\class\\train\\'+source),
                                 (test, str(Path.cwd())+'\\class\\test\\'+source)):
        for f in file_list:
            shutil.move(f, dirname)

def main():

    word = input('Scrivi cosa di cosa vuoi scaricare le immagini: ')   

    if not (os.path.exists(str(Path.cwd())+'\\class')):
        os.makedirs(str(Path.cwd())+'\\class')

    if not (os.path.exists(str(Path.cwd())+'\\tmp_'+word)):
        os.makedirs(str(Path.cwd())+'\\tmp_'+word)

    if not (os.path.exists(str(Path.cwd())+'\\class\\'+'\\train')):
        os.makedirs(str(Path.cwd())+'\\class\\'+'\\train')
    
    if not (os.path.exists(str(Path.cwd())+'\\class\\'+'\\train\\'+word)):
        os.makedirs(str(Path.cwd())+'\\class\\'+'\\train\\'+word)

    if not (os.path.exists(str(Path.cwd())+'\\class\\'+'\\test')):
        os.makedirs(str(Path.cwd())+'\\class\\'+'\\test')
    
    if not (os.path.exists(str(Path.cwd())+'\\class\\'+'\\test\\'+word)):
        os.makedirs(str(Path.cwd())+'\\class\\'+'\\test\\'+word)

    if not (os.path.exists(str(Path.cwd())+'\\class\\'+'\\validation')):
        os.makedirs(str(Path.cwd())+'\\class\\'+'\\validation')

    download_image_from_bing(word,n_image=60)
    make_train_and_test(word,0.8)
    os.rmdir(str(Path.cwd())+'\\tmp_'+word)
    
if __name__ == "__main__":
    main()
    
