import numpy as np


# https://github.com/davisvideochallenge/davis/blob/master/python/lib/davis/measures/jaccard.py
def eval_iou(annotation,segmentation):
    """ Compute region similarity as the Jaccard Index.

    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.

    Return:
        jaccard (float): region similarity

    """

    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)


# https://github.com/art-programmer/PlaneNet/blob/master/utils.py#L2115
def eval_plane_prediction(predSegmentations, gtSegmentations, predDepths, gtDepths, threshold=0.5):
    predNumPlanes = len(np.unique(predSegmentations)) - 1
    gtNumPlanes = len(np.unique(gtSegmentations)) - 1

    if len(gtSegmentations.shape) == 2:
        gtSegmentations = (np.expand_dims(gtSegmentations, -1) == np.arange(gtNumPlanes)).astype(np.float32)
    if len(predSegmentations.shape) == 2:
        predSegmentations = (np.expand_dims(predSegmentations, -1) == np.arange(predNumPlanes)).astype(np.float32)

    planeAreas = gtSegmentations.sum(axis=(0, 1))
    intersectionMask = np.expand_dims(gtSegmentations, -1) * np.expand_dims(predSegmentations, 2) > 0.5

    # depthDiffs = np.expand_dims(gtDepths, -1) - np.expand_dims(predDepths, 2)
    depthDiffs = gtDepths - predDepths
    depthDiffs = depthDiffs[:, :, np.newaxis, np.newaxis]

    intersection = np.sum((intersectionMask).astype(np.float32), axis=(0, 1))

    planeDiffs = np.abs(depthDiffs * intersectionMask).sum(axis=(0, 1)) / np.maximum(intersection, 1e-4)

    planeDiffs[intersection < 1e-4] = 1

    union = np.sum(((np.expand_dims(gtSegmentations, -1) + np.expand_dims(predSegmentations, 2)) > 0.5).astype(np.float32), axis=(0, 1))
    planeIOUs = intersection / np.maximum(union, 1e-4)

    numPredictions = int(predSegmentations.max(axis=(0, 1)).sum())

    numPixels = planeAreas.sum()

    IOUMask = (planeIOUs > threshold).astype(np.float32)
    minDiff = np.min(planeDiffs * IOUMask + 1000000 * (1 - IOUMask), axis=1)
    stride = 0.05
    pixelRecalls = []
    planeStatistics = []
    for step in range(int(0.61 / stride + 1)):
        diff = step * stride
        pixelRecalls.append(np.minimum((intersection * (planeDiffs <= diff).astype(np.float32) * IOUMask).sum(1),
                                       planeAreas).sum() / numPixels)
        planeStatistics.append(((minDiff <= diff).sum(), gtNumPlanes, numPredictions))

    return pixelRecalls, planeStatistics


#https://github.com/art-programmer/PlaneNet
def evaluateDepths(predDepths, gtDepths, validMasks, planeMasks=True, printInfo=True):
    masks = np.logical_and(np.logical_and(validMasks, planeMasks), gtDepths > 1e-4)

    numPixels = float(masks.sum())

    rmse = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)
    rmse_log = np.sqrt((pow(np.log(predDepths) - np.log(gtDepths), 2) * masks).sum() / numPixels)
    log10 = (np.abs(
        np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels
    rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    rel_sqr = (pow(predDepths - gtDepths, 2) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (
            1 - masks.astype(np.float32)) * 10000
    accuracy_1 = (deltas < 1.25).sum() / numPixels
    accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
    accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels
    recall = float(masks.sum()) / validMasks.sum()
    if printInfo:
        print(('evaluate', rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3, recall))
        pass
    return rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3, recall


def eval_plane_and_pixel_recall_normal(segmentation, gt_segmentation, param, gt_param, threshold=0.5):
    """
    :param segmentation: label map for plane segmentation [h, w] where 20 indicate non-planar
    :param gt_segmentation: ground truth label for plane segmentation where 20 indicate non-planar
    :param threshold: value for iou
    :return: percentage of correctly predicted ground truth planes correct plane
    """
    depth_threshold_list = np.linspace(0.0, 30, 13)

    # both prediction and ground truth segmentation contains non-planar region which indicated by label 20
    # so we minus one
    plane_num = len(np.unique(segmentation)) - 1
    gt_plane_num = len(np.unique(gt_segmentation)) - 1

    # 13: 0:0.05:0.6
    plane_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))
    pixel_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))

    plane_area = 0.0

    gt_param = gt_param.reshape(20, 3)

    # check if plane is correctly predict
    for i in range(gt_plane_num):
        gt_plane = gt_segmentation == i
        plane_area += np.sum(gt_plane)

        for j in range(plane_num):
            pred_plane = segmentation == j
            iou = eval_iou(gt_plane, pred_plane)

            if iou > threshold:
                # mean degree difference over overlap region:
                gt_p = gt_param[i]
                pred_p = param[j]

                n_gt_p = gt_p / np.linalg.norm(gt_p)
                n_pred_p = pred_p / np.linalg.norm(pred_p)

                angle = np.arccos(np.clip(np.dot(n_gt_p, n_pred_p), -1.0, 1.0))
                degree = np.degrees(angle)
                depth_diff = degree

                # compare with threshold difference
                plane_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32)
                pixel_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32) * \
                      (np.sum(gt_plane * pred_plane))
                break

    pixel_recall = np.sum(pixel_recall, axis=0).reshape(1, -1) / plane_area

    return plane_recall, pixel_recall

