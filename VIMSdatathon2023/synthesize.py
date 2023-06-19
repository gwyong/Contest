import os, glob, random
import cv2
import numpy as np

from PIL import Image
from rembg import remove

def get_foreground(foreground_path):
    save_dir = foreground_path.split(".jpg")[0] + "_BG_removed.png"
    output = remove(Image.open(foreground_path))
    output.save(save_dir)
    print("finished.")

def synthesize_from_background(background_path, foreground_path, save_dir):
    background_img = cv2.cvtColor(cv2.imread(background_path), cv2.COLOR_BGR2RGB).copy()
    foreground_img = cv2.cvtColor(cv2.imread(foreground_path), cv2.COLOR_BGR2RGB).copy()

    bg_H, bg_W, _ = background_img.shape
    fg_H, fg_W, _ = foreground_img.shape

    # resize foreground to background
    scaling_factor = min(bg_H / fg_H, bg_W / fg_W)
    resized_foreground = cv2.resize(foreground_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    # resize foreground much smaller
    scaling_factor = round(random.uniform(1/3, 3/4), 2)
    resized_foreground = cv2.resize(resized_foreground, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    h, w, _ = resized_foreground.shape
    x = random.randint(0, bg_W - w -1)
    y = random.randint(0, bg_H - h -1)

    mask_boolean = np.all(resized_foreground == [0, 0, 0], axis=2)
    mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)
    
    background_img[y:y+h, x:x+w, :] = background_img[y:y+h, x:x+w, :]*mask_rgb_boolean + (resized_foreground)

    cv2.imwrite(save_dir, cv2.cvtColor(background_img, cv2.COLOR_RGB2BGR))
    print("Synthesized completely!")

def find_class_from_path(path, class_dict):
    if "crane" in path:
        id = class_dict["crane"]
    if "excavator" in path:
        id = class_dict["excavator"]
    if "bulldozer" in path:
        id = class_dict["bulldozer"]
    if "scraper" in path:
        id = class_dict["scraper"]
    if "truck" in path:
        id = class_dict["truck"]
    if "worker" in path:
        id = class_dict["worker"]
    return id
    
class Cut_and_Paste():
    def __init__(self, save_image_dir=None, save_label_dir=None):
        self.save_image_dir = save_image_dir
        self.save_label_dir = save_label_dir
    
    def get_foreground(self, foreground_path):
        save_dir = foreground_path.split(".jpg")[0] + "_BG_removed.png"
        output = remove(Image.open(foreground_path))
        output.save(save_dir)
        print("Extracted foreground objects.")

    def synthesize(self, background_path, foreground_folder_path, filename, is_flip=True, K=3):
        background_img = cv2.cvtColor(cv2.imread(background_path), cv2.COLOR_BGR2RGB)
        foreground_paths = glob.glob(os.path.join(foreground_folder_path, "*.png"))
        foreground_paths = random.sample(foreground_paths, K)
        bg_H, bg_W, _ = background_img.shape
        
        yolov8_labels= []
        class_dict = {'crane':0,
                      'excavator':1,
                      'bulldozer':2,
                      'scraper':3,
                      'truck':4,
                      'worker':5,
                      }
        
        for foreground_path in foreground_paths:
            foreground_img = cv2.cvtColor(cv2.imread(foreground_path), cv2.COLOR_BGR2RGB)
            fg_H, fg_W, _ = foreground_img.shape

            # initally resize foreground to background for not distorting foregroundg images
            scaling_factor = min(bg_H / fg_H, bg_W / fg_W)
            resized_foreground = cv2.resize(foreground_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            
            # resize foreground much smaller
            scaling_factor = round(random.uniform(1/3, 3/4), 2)
            resized_foreground = cv2.resize(resized_foreground, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

            if is_flip:
                resized_foreground = cv2.flip(resized_foreground, 1)
            
            fg_h, fg_w, _ = resized_foreground.shape
            x = random.randint(0, bg_W - fg_w -1)
            y = random.randint(0, bg_H - fg_h -1)
            cx = (x + fg_w/2)/bg_W # including normalization
            cy = (y + fg_h/2)/bg_H # including normalization
            
            cx = format(cx, ".6f")
            cy = format(cy, ".6f")
            nw = format(fg_w/bg_W, ".6f")
            nh = format(fg_h/bg_H, ".6f")
            cid = find_class_from_path(foreground_path, class_dict)
            yolov8_labels.append((int(cid), cx, cy, nw, nh))
            
            mask_boolean = np.all(resized_foreground == [0, 0, 0], axis=2)
            mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)
            background_img[y:y+fg_h, x:x+fg_w, :] = background_img[y:y+fg_h, x:x+fg_w, :]*mask_rgb_boolean + (resized_foreground)
        
        save_label_dir = os.path.join(self.save_label_dir, filename) + ".txt"
        with open(save_label_dir, 'w') as f:
            for info in yolov8_labels:
                aLine = ' '.join(map(str,info))
                f.write(aLine+'\n')
        save_dir = os.path.join(self.save_image_dir, filename) + ".jpg"
        cv2.imwrite(save_dir, cv2.cvtColor(background_img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    random.seed(0)
    foreground_path = "./Foreground"
    background_path = "./Background"
    save_image_path = "./Synthesized Images"
    save_label_path = "./Synthesized Labels"
    
    jpg_format = "*.jpg"
    png_format = "*.png"
    CP = Cut_and_Paste(save_image_dir=save_image_path, save_label_dir=save_label_path)
    
    # foreground_paths = glob.glob(os.path.join(foreground_path, jpg_format))
    # for foreground_path in foreground_paths:
    #     CP.get_foreground(foreground_path)

    # foreground_paths = glob.glob(os.path.join(foreground_path, png_format))
    background_paths = glob.glob(os.path.join(background_path, jpg_format))
    num_synthesize = 20

    for i, bg_path in enumerate(background_paths):
        for n in range(num_synthesize):
            filename = str(i) + "_" + str(n)
            # print(i, n, "|", len(background_paths), num_synthesize)
            K = random.randint(1, 3)
            is_flip = random.sample([True, False], 1)[0]
            CP.synthesize(bg_path, foreground_path, filename, is_flip=is_flip, K=K)
    
    print("All completed.")