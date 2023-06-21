import json, time
import numpy as np
import pandas as pd
from tqdm import tqdm

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def iou_oa(y_true, y_pred, n_class):
    iou = []

    # calculate iou per class
    tp = 0
    for c in range(n_class+1): # +1 for the background
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))

        n = TP + 1e-12
        d = float(TP + FP + FN + 1e-12)
        if c != n_class:
          iou.append(np.divide(n, d))
        tp += TP

    oa =  tp / len(y_true)
    return (np.mean(iou), oa, iou[0], iou[1], iou[2], iou[3], iou[4], iou[5])

def compute_metrics(coco_true_json, coco_pred_json):
    image_paths = [item["file_name"] for item in coco_true_json["images"]]
    start_time = time.time()
    m_iou_list, oa_list, iou_0_list, iou_1_list, iou_2_list, iou_3_list, iou_4_list, iou_5_list = [], [], [], [], [], [], [], []

    for filepath in tqdm(image_paths, desc="Computing metrics..."):
        filename = filepath.split(".")[0]
        images = coco_true_json["images"]
        for image in images:
            if image["file_name"].split(".")[0] == filename:
                W, H = image["width"], image["height"]
                break
    
        nc = 6
        
        ## Get y_true
        y_true = np.ones((W, H, nc))*nc # nc index will be the background
        annotations = coco_true_json["annotations"]
        for annotation in annotations:
            image_id = annotation["image_id"]
            if image_id == filename:
                x, y, w, h = annotation["bbox"]
                x_start = int(x)
                x_end = int(x+w)
                y_start = int(y)
                y_end = int(y+h)
                category_index = annotation["category_id"]
                y_true[x_start:x_end, y_start:y_end, category_index] = category_index 

        ## Get y_pred
        y_pred = np.ones((W, H, nc))*nc
        for pred_val in coco_pred_json:
            image_id = pred_val["file_name"].split(".")[0]
            if image_id == filename:
                x, y, w, h = pred_val["bbox"]
                x_start = int(x)
                x_end = int(x+w)
                y_start = int(y)
                y_end = int(y+h)
                category_index = pred_val["category_id"]
                y_pred[x_start:x_end, y_start:y_end, category_index] = category_index
        
        m_iou, oa, iou_0, iou_1, iou_2, iou_3, iou_4, iou_5 = iou_oa(y_true.flatten(), y_pred.flatten(), nc)
        m_iou_list.append(m_iou)
        oa_list.append(oa)
        iou_0_list.append(iou_0)
        iou_1_list.append(iou_1)
        iou_2_list.append(iou_2)
        iou_3_list.append(iou_3)
        iou_4_list.append(iou_4)
        iou_5_list.append(iou_5)
    
    end_time = time.time()
    print("Elapsed Time:", end_time-start_time)
    
    df = pd.DataFrame({"IMAGE PATHS": image_paths,
                       "MEAN IOU": m_iou_list,
                       "OA": oa_list,
                       "ClS 0 IOU": iou_0_list,
                       "ClS 1 IOU": iou_1_list,
                       "ClS 2 IOU": iou_2_list,
                       "ClS 3 IOU": iou_3_list,
                       "ClS 4 IOU": iou_4_list,
                       "ClS 5 IOU": iou_5_list
                       })
    averages = df.mean()
    print("Averages:")
    print(averages)
    df.loc[df.shape[0]] = averages
    df.at[df.shape[0] - 1, 'IMAGE PATHS'] = 'AVERAGE'
    df.to_csv('metrics.csv', index=False)
    return