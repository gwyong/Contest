import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt_path = "./gt_annotation_coco.json"
dt_path = "./dt_annotation_coco.json"

gt_coco = COCO(gt_path)
dt_coco = gt_coco.loadRes(dt_path)

cocoEval = COCOeval(gt_coco, dt_coco, iouType='bbox')
# cocoEval.params.imgIds = [1, 2, 3, 4, ..., 121]
# cocoEval.params.catIds = [1, 2, 3, 4, 5, 6]
# cocoEval.params.maxDets = [1, 10, 100]
# cocoEval.params.areaRng = [[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]
# cocoEval.params.iouThrs = [0.5  0.55 0.6  0.65 0.7  0.75 0.8  0.85 0.9  0.95]
# cocoEval.params.recThrs = [0., 0.01, ... 0.99, 1.]
# cocoEval.eval["precision"].shape = (10, 101, 6, 4, 3) -> (T,R,K,A,M) -> (len(iouThrs), len(recThrs), len(catIds), len(areaRng), len(maxDets))
# cocoEval.eval["recall"].shape = (10, 6, 4, 3)

# evaluate detections
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

def check_precision_recall_per_class(ap=1, iouThr=None, areaRng="all", maxDets=cocoEval.params.maxDets[2], num_class=6):
    """
    input args:
        ap: use average precision or not(0)
        iouThr: None or [0.5  0.55 0.6  0.65 0.7  0.75 0.8  0.85 0.9  0.95]
    """
    p = cocoEval.params
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = cocoEval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,:,aind,mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = cocoEval.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,aind,mind]
    if len(s[s>-1])==0:
        print("-1 returning")
        mean_s = -1
    else:
        try:
            s_ = s[s>-1].reshape((10, 101, num_class, 1))
        except:
            try:
                s_ = s[s>-1].reshape((1, 101, num_class, 1))
            except:
                s_ = s[s>-1].reshape((10, num_class, 1))
        print("This is per class precision/recall value.")
        print(s_)

def count_instance(gt_path, dt_path):
    with open(gt_path, 'r') as f:
        data = json.load(f)

    class_counts = {}

    # Iterate over the annotations and count instances for each class
    for annotation in data['annotations']:
        category_id = annotation['category_id']
        class_name = next(category['name'] for category in data['categories'] if category['id'] == category_id)
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1

    print("The number of instances in the ground truth COCO json file.")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
    print("="*10)

    with open(dt_path, 'r') as f:
        data = json.load(f)
    
    class_counts = {}
    with open(gt_path, 'r') as f:
        ground_truth_data = json.load(f)
        category_mapping = {category['id']: category['name'] for category in ground_truth_data['categories']}

    # Iterate over the predicted annotations and count instances for each class
    for annotation in data:
        category_id = annotation['category_id']
        class_name = category_mapping.get(category_id, 'Unknown')
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1

    print("The number of instances in the detection result COCO json file.")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")