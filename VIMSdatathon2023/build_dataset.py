import os, glob, shutil, random, time
from PIL import Image, ImageFile
from collections import Counter
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_folders_in_dir(dir):
    return [item for item in os.listdir(dir) if os.path.isdir(os.path.join(dir, item))]

def assert_lists_equal_ignore_order(list_a, list_b, error_msg):
    assert Counter(list_a) == Counter(list_b), error_msg

def make_YOLOv8_data_folder():
    if os.path.exists("./train"):
        shutil.rmtree("./train")
    if os.path.exists("./test"):
        shutil.rmtree("./test")
    os.mkdir("./train") 
    os.mkdir("./test")       
    os.mkdir("./train/images")
    os.mkdir("./train/labels")
    os.mkdir("./test/images")
    os.mkdir("./test/labels")

def join_checked_frames_and_labels(checked_frames_dir, checked_labels_dir, jpeg=False):
    assert_lists_equal_ignore_order(get_folders_in_dir(checked_frames_dir),
                                    get_folders_in_dir(checked_labels_dir),
                                    error_msg="Checked Frames != Checked Labels")
    
    video_names = get_folders_in_dir(checked_frames_dir)
    images, labels = [], []
    for video_name in video_names:
        images.extend(glob.glob(os.path.join(checked_frames_dir, video_name, "*.jpg")))
        if jpeg:
            images.extend(glob.glob(os.path.join(checked_frames_dir, video_name, "*.jpeg")))
        labels.extend(glob.glob(os.path.join(checked_labels_dir, video_name, "*.txt")))
    return images, labels

def find_savotage(dir_frame, dir_label):
    image_filenames = glob.glob(os.path.join(dir_frame, "*.jpg"))
    label_filenames = glob.glob(os.path.join(dir_label, "*.txt"))
    
    img_frames = []
    lab_frames = []
    
    for filename in image_filenames:
        filename = filename.split("_")[-1].split(".")[0]
        img_frames.append(filename)

    for filename in label_filenames:
        filename = filename.split("_")[-1].split(".")[0]
        lab_frames.append(filename)
    
    for frame in img_frames:
        if frame not in lab_frames:
            print(frame) 
    
def save_image_to_path(image_path, save_path):
    Image.open(image_path).save(save_path)

def copy_label_file(label_path, save_path):
    shutil.copy2(label_path, save_path)

def split_train_test(image_paths, label_paths, ratio=0.9):
    assert len(image_paths) == len(label_paths), "Number of paths in images and labels did not match."

    data_paths = []
    for img_path, lab_path in zip(image_paths, label_paths):
        data_paths.append((img_path, lab_path))

    K = int(len(data_paths)*ratio)
    train = random.sample(data_paths, K)
    test = [item for item in data_paths if item not in train]
    print("train:test %d:%d", (len(train), len(test)))

    train_image_dir = "./train/images"
    train_label_dir = "./train/labels"
    test_image_dir = "./test/images"
    test_label_dir = "./test/labels"

    for img_dir, lab_dir in tqdm(train, desc="preparing training set..."):
        time.sleep(0.01)
        img_name = img_dir.split("\\")[-2] + "_" + img_dir.split("\\")[-1].split(".")[0] + ".jpg"
        lab_name = lab_dir.split("\\")[-2] + "_" + lab_dir.split("\\")[-1]
        new_img_dir = os.path.join(train_image_dir, img_name)
        new_lab_dir = os.path.join(train_label_dir, lab_name)
        save_image_to_path(img_dir, new_img_dir)
        copy_label_file(lab_dir, new_lab_dir)

    for img_dir, lab_dir in tqdm(test, desc="preparing test set..."):
        time.sleep(0.01)
        img_name = img_dir.split("\\")[-2] + "_" + img_dir.split("\\")[-1].split(".")[0] + ".jpg"
        lab_name = lab_dir.split("\\")[-2] + "_" + lab_dir.split("\\")[-1]
        new_img_dir = os.path.join(test_image_dir, img_name)
        new_lab_dir = os.path.join(test_label_dir, lab_name)
        save_image_to_path(img_dir, new_img_dir)
        copy_label_file(lab_dir, new_lab_dir)
    
    print("Train Test Split Completed!")

if __name__ == "__main__":
    dir_worker = "./worker" # co-worker's work directory
    make_YOLOv8_data_folder()

    images, labels = [], []
    
    imgs, labs = join_checked_frames_and_labels("./Checked Frames","./Checked Labels")
    images.extend(imgs); labels.extend(labs)
    
    imgs, labs = join_checked_frames_and_labels(
        os.path.join(dir_worker, get_folders_in_dir(dir_worker)[0]),
        os.path.join(dir_worker, get_folders_in_dir(dir_worker)[1]),
    )
    images.extend(imgs); labels.extend(labs)
    
    split_train_test(images, labels, ratio=0.8)