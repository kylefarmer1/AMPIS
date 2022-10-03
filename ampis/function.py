"""
A Collection of analysis and helper functions used in the evaluation notebooks 
for analysis of the instance segmentation model geared toward regions hidden from view.

"""

from ampis.structures import InstanceSet, RLEMasks, masks_to_rle, masks_to_bitmask_array
import numpy as np
import pandas as pd
from ampis.structures import mask_areas
import torch
import pycocotools.mask as rle
from ampis.analyze import merge_boxes
from skimage.morphology import binary_erosion


def mask_edge_distance_overlap(gt_mask, pred_mask, gt_box, pred_box, matches, device='auto'):
    """
    Investigate the disagreement between the boundaries of predicted and ground truth masks.

    For every matched pair of masks in pred and gt, determine false positive and false negative pixels.
    For every false positive pixel, compute the distance to the nearest ground truth pixel.
    For every false negative pixel, compute the distance to the nearest predicted pixel.

    Parameters
    -------------
    gt_mask, pred_mask: list or RLEMasks
        ground truth and predicted masks, RLE format
    gt_box, pred_box: array
        array of bbox coordinates
    matches: array
        n_match x 2 element array where matches[i] gives the index of the ground truth and predicted masks in
        gt and pred corresponding to match i. This can be obtained from mask_match_stats (results['match_tp'])
    device: str
        Determines which device is used.
        'cpu': cpu
        'cuda': CUDA-compatible gpu, if available, otherwise cpu will be used.


    Returns
    -------------
    FP_distances, FN_distances: list(torch.tensor)
        List of results for each match in matches. Each element is a tensor containing the euclidean distances
        (in pixels) from each false positive to its nearest ground truth pixel(FP_distances)
        or the distance from each false negative to the nearest predicted pixel(FN_distances).
    """

    if device.lower() == 'cuda' and torch.cuda.is_available():  # handle case where gpu is selected but unavailable
        device = 'cuda'
    else:
        device = 'cpu'

    if type(gt_mask) == RLEMasks:
        gt_mask = gt_mask.rle
    if type(pred_mask) == RLEMasks:
        pred_mask = pred_mask.rle

    gt_masks = [gt_mask[i] for i in matches[:, 0]]
    gt_boxes = [gt_box[i] for i in matches[:, 0]]

    pred_masks = [pred_mask[i] for i in matches[:, 1]]
    pred_boxes = [pred_box[i] for i in matches[:, 1]]

    t = np.zeros((len(gt_mask),*gt_mask[0]['size']))
    for idx,mask in enumerate(gt_mask):
        t[idx,:,:]=rle.decode(mask)


    FP_distances = []
    FN_distances = []
    FP_indices = []
    FN_indices = []
    for gm, pm, gb, pb, m in zip(gt_masks, pred_masks, gt_boxes, pred_boxes, matches):
        # masks are same size as whole image, but we only need to look in the region containing the masks.
        # combine the bboxes to get region containing both masks
        c1, r1, c2, r2 = np.round(merge_boxes(gb, pb)).astype(int)

        # decode RLE, select subset of masks included in box, and cast to torch tensor
        g = torch.tensor(rle.decode(gm)[r1:r2, c1:c2], dtype=torch.bool).to(device)
        p = torch.tensor(rle.decode(pm)[r1:r2, c1:c2], dtype=torch.bool).to(device)

        # indices of pixels included in ground truth and predicted masks
        gt_where = torch.stack(torch.where(g), axis=1)
        pred_where = torch.stack(torch.where(p), axis=1)

        # indices of false positive (pred and not gt) and false negative (gt and not pred) pixels
        FP_where = torch.stack(torch.where(p & torch.logical_not(g)), axis=1)
        FN_where = torch.stack(torch.where(g & torch.logical_not(p)), axis=1)

        # compile ground truths without current gt mask
        t_drop = np.delete(t,m[0],0)

        # collaps roi on ground truths
        t_roi = torch.tensor(np.logical_or.reduce(t_drop,axis=0)[r1:r2,c1:c2],dtype=torch.bool).to(device)

        # indices of false positive overlapping with ground truth
        FP_O_where = torch.where(p[FP_where[:,0],FP_where[:,1]] & t_roi[FP_where[:,0],FP_where[:,1]])[0]

        # indices of false negative overlapping with ground truth
        FN_O_where = torch.where(g[FN_where[:,0],FN_where[:,1]] & t_roi[FN_where[:,0],FN_where[:,1]])[0]

        # distance from false positives to nearest ground truth pixels
        if FP_where.numel():
            FP_dist = _min_euclid(FP_where, gt_where)

        else:
            FP_dist = torch.tensor([], dtype=torch.double)

        # distance from false negatives to nearest predicted pixels
        if FN_where.numel():
            FN_dist = _min_euclid(FN_where, pred_where)
        else:
            FN_dist = torch.tensor([], dtype=torch.double)

        FP_distances.append(FP_dist)
        FN_distances.append(FN_dist)
        FP_indices.append(FP_O_where)
        FN_indices.append(FN_O_where)

    return FP_distances, FN_distances, FP_indices, FN_indices



def parse_overlapping_distances(distances,indices):
    # func that separates distances relating to pixels that have an overlapping ground truth
    # input: distances: list of arrays
    #        indices: list of arrays
    # returns: do: list of arrays (corresponding to overlapping regions)
    #          dn: list of ararys (corresponding to non-overlapping regions)
    do = []
    dn = []
    for I,D in zip(indices,distances):
        do.append(D[I])
        dn.append(np.delete(D,I))
        
    return do,dn


def cumsum(distances):
    # func that returns the x and y values of a cumulative distribution function
    # input: distances (list of arrays)
    unique,counts = np.unique(np.concatenate(distances),return_counts=True)
    counts = counts.cumsum()
    counts = counts/counts[-1]
    return unique, counts


def partition_overlaps_pred(iset,gt_iset):
    
    """
    Partitions an instance set into regions of overlap (with gt) and nonoverlap
    
    Parameters
    ----------
    iset: InstanceSet to partition
    gt_iset: ground truth InstanceSet. Used for determining overlaps (can be the same as
                iset if partition the ground truth iset)
    
    
    Returns
    --------
    iset_o: InstanceSet
        Instance containing masks witih only overlapping pixels
    iset_no: InstanceSet
        Instance containing masks with only non overlapping pixels
    
    """
    # make copies of the instance set
    iset_o = iset.copy()
    iset_no = iset.copy()
    
    boxes = iset.instances.boxes
    masks = iset.instances.masks.rle
    gt_masks = gt_iset.instances.masks.rle
    
    o_boxes = list()
    no_boxes = list()
    o_masks = list()
    no_masks = list()
    
    # get all the ground truth masks into a single object to collapse later
    t = np.zeros((len(gt_masks),*gt_masks[0]['size']))
    for idx,mask in enumerate(gt_masks):
        t[idx,:,:]=rle.decode(mask)
    
    # compare each mask in the ground truth to the collapse image, then compare with 
    # logical_and
    overlaps = np.zeros((len(gt_masks), *gt_masks[0]['size']))
    for idx,mask in enumerate(gt_masks):
        t_drop = np.delete(t,idx,0)
        t_roi = np.logical_or.reduce(t_drop,axis=0)
        g_o = np.logical_and(mask,t_roi)
        overlaps[idx,:,:]=g_o

    # collapse all overlapping regions
    t_roi = np.logical_or.reduce(overlaps,axis=0)

    # iterate through the masks and boxes to partition pixels that are in overlaps
    for idx, (box,mask) in enumerate(tqdm(zip(boxes,masks),total=len(masks))):
        # remove the current mask from the collection then collapse
        mask = rle.decode(mask)

        # partition into overlapping and non-overlapping
        o = np.logical_and(mask,t_roi)
        no = np.logical_and(mask,np.logical_xor(mask,t_roi))

        # find new bounding boxes (c1,r1,c2,r2)
        o_box = _extract_box(o)
        no_box = _extract_box(no)

        # append new partitions
        o_boxes.append(o_box)
        no_boxes.append(no_box)

        # convert back to rlemask
        o_masks.append(rle.encode(np.asfortranarray(o)))
        no_masks.append(rle.encode(np.asfortranarray(no)))

    # add to instance sets
    iset_o.instances.boxes=o_boxes
    iset_no.instances.boxes=no_boxes
    iset_o.instances.masks=RLEMasks(o_masks)
    iset_no.instances.masks=RLEMasks(no_masks)
    
    return iset_o, iset_no

def _parse_dss(scores):
    # helper function to get all the scores
    return scores['det_tp'], scores['det_fn'], scores['det_fp'], scores['seg_precision'], scores['seg_recall']

def get_dss_stats(results):
    dtp, dfn, dfp, sp, sr = zip(*[_parse_dss(scores) for scores in results])

    merged_dtp = np.vstack(dtp)
    merged_dfn = np.hstack(dfn)
    merged_dfp = np.hstack(dfp)

    total_precision = len(merged_dtp)/(len(merged_dfn)+len(merged_dtp))
    total_recall = len(merged_dtp)/(len(merged_dfp)+len(merged_dtp))

    total_seg_precision = np.hstack(sp)
    total_seg_recall = np.hstack(sr)
    total_seg_precision = total_seg_precision[~np.isnan(total_seg_precision)]
    total_seg_recall = total_seg_recall[~np.isnan(total_seg_recall)]

    median_seg_precision = np.median(total_seg_precision)
    median_seg_recall = np.median(total_seg_recall)

    mean_seg_precision = np.mean(total_seg_precision)
    mean_seg_recall = np.mean(total_seg_recall)

    return {'total_det_precision': total_precision,
            'total_det_recall': total_recall,
            'median_seg_recall': median_seg_recall,
            'median_seg_precision': median_seg_precision,
            'mean_seg_recall':mean_seg_recall,
            'mean_seg_precision':mean_seg_precision,
            'std_seg_precision':total_seg_precision.std(),
            'std_seg_recall':total_seg_recall.std()}

def print_dss_stats(scores, return_metrics=False):
    metrics = get_dss_stats(scores)
    s=f"\t detection precision: {metrics['total_det_precision']:.3f} \n \
        detection recall: {metrics['total_det_recall']:.3f}  \n \
        median seg. precision: {metrics['median_seg_precision']:.3f} \n \
        median seg. recall: {metrics['median_seg_recall']:.3f} \n \
        mean seg. precision: {metrics['mean_seg_precision']:.3f} \n \
        mean seg. recall: {metrics['mean_seg_recall']:.3f} \n \
        std. dev. seg. precision: {metrics['std_seg_precision']:.3f} \n \
        std. dev. seg. recall: {metrics['std_seg_recall']:.3f}"
    print(s)
    if return_metrics:
        return metrics

def _extract_box(mask):
    if not np.any(mask):
        return np.array([])
    else:
        horizontal = np.where(np.any(mask,axis=0))[0]
        vertical = np.where(np.any(mask,axis=0))[0]
        c1,c2=horizontal[[0,-1]]
        r1,r2=vertical[[0,-1]]
        return c1,r1,c2,r2

def partition_overlaps(iset,gt_iset):
    
    """
    Partitions an instance set into regions of overlap (with gt) and nonoverlap
    
    Parameters
    ----------
    iset: InstanceSet to partition
    gt_iset: ground truth InstanceSet. Used for determining overlaps (can be the same as
                iset if partition the ground truth iset)
    
    
    Returns
    --------
    iset_o: InstanceSet
        Instance containing masks witih only overlapping pixels
    iset_no: InstanceSet
        Instance containing masks with only non overlapping pixels
    
    """
    # make copies of the instance set
    iset_o = iset.copy()
    iset_no = iset.copy()
    
    
    boxes = iset.instances.boxes
    masks = iset.instances.masks.rle
    gt_masks = gt_iset.instances.masks.rle
    
    o_boxes = list()
    no_boxes = list()
    o_masks = list()
    no_masks = list()
    
    # get all the ground truth masks into a single object to collapse later
    t = np.zeros((len(gt_masks),*gt_masks[0]['size']))
    for idx,mask in enumerate(gt_masks):
        t[idx,:,:]=rle.decode(mask)
    
    # compare each mask in the ground truth to the collapse image, then compare with 
    # logical_and
    overlaps = np.zeros((len(gt_masks), *gt_masks[0]['size']))
    for idx,mask in enumerate(gt_masks):
        t_drop = np.delete(t,idx,0)
        t_roi = np.logical_or.reduce(t_drop,axis=0)
        g_o = np.logical_and(rle.decode(mask),t_roi)
        overlaps[idx,:,:]=g_o

    # collapse all overlapping regions
    t_roi = np.logical_or.reduce(overlaps,axis=0)

    # iterate through the masks and boxes to partition pixels that are in overlaps
    for idx, (box,mask) in enumerate(zip(boxes,masks)):
        # remove the current mask from the collection then collapse
        mask = rle.decode(mask)

        # partition into overlapping and non-overlapping
        o = np.logical_and(mask,t_roi)
        no = np.logical_and(mask,np.logical_xor(mask,t_roi))

        # find new bounding boxes (c1,r1,c2,r2)
        o_box = _extract_box(o)
        no_box = _extract_box(no)

        # append new partitions
        o_boxes.append(o_box)
        no_boxes.append(no_box)

        # convert back to rlemask
        o_masks.append(rle.encode(np.asfortranarray(o)))
        no_masks.append(rle.encode(np.asfortranarray(no)))

    # add to instance sets
    iset_o.instances.boxes=o_boxes
    iset_no.instances.boxes=no_boxes
    iset_o.instances.masks=RLEMasks(o_masks)
    iset_no.instances.masks=RLEMasks(no_masks)
    
    return iset_o, iset_no

def group_areas_by_distribution(gt,pred):
    '''
    Helper function that will group areas from masks by their distribution
    parameters:
        gt: InstanceSet
        pred: InstanceSet
        
    returns:
        gt_areas_gr: dict
        pred_areas_gr: dict
    '''
    gt_areas=dict()
    pred_areas=dict()
    for i, (gt, pred) in enumerate(zip(gt, pred)):
        gt_areas[gt.filepath.stem]=mask_areas(gt)
        pred_areas[pred.filepath.stem]=mask_areas(pred)

    gt_areas_df= pd.DataFrame(gt_areas.items(),columns=['filename','areas'])
    gt_areas_df['dist_class']=gt_areas_df['filename'].str[0]

    pred_areas_df= pd.DataFrame(pred_areas.items(),columns=['filename','areas'])
    pred_areas_df['dist_class']=pred_areas_df['filename'].str[0]

    gt_areas_gr = dict()
    for char in gt_areas_df['dist_class'].unique():
        gt_areas_gr[char]=np.concatenate(gt_areas_df[gt_areas_df['dist_class']==char]['areas'].values)
    pred_areas_gr = dict()
    for char in pred_areas_df['dist_class'].unique():
        pred_areas_gr[char]=np.concatenate(pred_areas_df[pred_areas_df['dist_class']==char]['areas'].values)

    return gt_areas_gr, pred_areas_gr


##########################################
####    Boundary analysis functions   ####
##########################################
def lighten_background(image,replace=[31,31,31]):
    te = image.copy()
    r,g,b = image[:,:,0], image[:,:,1], image[:,:,2]
    mask = (r == 0) & (g==0) & (b==0)
    te[:,:,:3][mask] = [31,31,31]
    return te

def _min_euclid(a, b):
    a = a.unsqueeze(axis=1)

    square_diffs = torch.pow(a.double() - b.double(), 2)

    distances = torch.sqrt(square_diffs.sum(axis=2))

    min_distances = distances.min(axis=1)[0]

    return min_distances

def boundary_f_score(gt_mask, pred_mask, gt_box, pred_box, matches, theta, device='cpu'):
    
    if device.lower() == 'cuda' and torch.cuda.is_available():  # handle case where gpu is selected but unavailable
        device = 'cuda'
    else:
        device = 'cpu'

    if type(gt_mask) == RLEMasks:
        gt_mask = gt_mask.rle
    if type(pred_mask) == RLEMasks:
        pred_mask = pred_mask.rle

    gt_masks = [gt_mask[i] for i in matches[:, 0]]
    gt_boxes = [gt_box[i] for i in matches[:, 0]]

    pred_masks = [pred_mask[i] for i in matches[:, 1]]
    pred_boxes = [pred_box[i] for i in matches[:, 1]]
    
    BF_scores = list()
    for gm, pm, gb, pb, m in zip(gt_masks, pred_masks, gt_boxes, pred_boxes, matches):
        # masks are same size as whole image, but we only need to look in the region containing the masks.
        # combine the bboxes to get region containing both masks
        c1, r1, c2, r2 = merge_boxes(gb, pb).astype(int)

        # decode RLE, select subset of masks included in box, and cast to torch tensor
        g = torch.tensor(rle.decode(gm)[r1:r2, c1:c2], dtype=torch.bool).to(device)
        p = torch.tensor(rle.decode(pm)[r1:r2, c1:c2], dtype=torch.bool).to(device)
        g = torch.nn.ConstantPad2d(1,0)(g)
        p = torch.nn.ConstantPad2d(1,0)(p)
        
        B_g = g^binary_erosion(g)
        B_p = p^binary_erosion(p)
        
        # indices of pixels included in ground truth and predicted masks
        gt_where = torch.stack(torch.where(B_g), axis=1)
        pred_where = torch.stack(torch.where(B_p), axis=1)
        
        g_dist = _min_euclid(gt_where,pred_where)
        p_dist = _min_euclid(pred_where,gt_where)
        
        Pc = 1/pred_where.shape[0]*(p_dist[p_dist<theta].shape[0])
        Rc = 1/gt_where.shape[0]*(g_dist[g_dist<theta].shape[0])
        
        if Pc+Rc==0:
            BF_scores.append(0)
        else:
            F1 = 2*Pc*Rc/(Pc+Rc)
            BF_scores.append(F1)
        
    BF_mean = np.mean(BF_scores)
        
        
    return BF_scores


def boundary_f_score_overlap(gt_mask, pred_mask, gt_box, pred_box, matches, theta, device='cpu'):

    if device.lower() == 'cuda' and torch.cuda.is_available():  # handle case where gpu is selected but unavailable
        device = 'cuda'
    else:
        device = 'cpu'

    if type(gt_mask) == RLEMasks:
        gt_mask = gt_mask.rle
    if type(pred_mask) == RLEMasks:
        pred_mask = pred_mask.rle

    gt_masks = [gt_mask[i] for i in matches[:, 0]]
    gt_boxes = [gt_box[i] for i in matches[:, 0]]

    pred_masks = [pred_mask[i] for i in matches[:, 1]]
    pred_boxes = [pred_box[i] for i in matches[:, 1]]

    # Store ground truth masks in binarized format
    t = np.zeros((len(gt_mask),*gt_mask[0]['size']))
    for idx,mask in enumerate(gt_mask):
        t[idx,:,:]=rle.decode(mask)

    BF_scores_o = list()
    BF_scores_no = list()
    for gm, pm, gb, pb, m in zip(gt_masks, pred_masks, gt_boxes, pred_boxes, matches):
        # masks are same size as whole image, but we only need to look in the region containing the masks.
        # combine the bboxes to get region containing both masks
        c1, r1, c2, r2 = merge_boxes(gb, pb).astype(int)

        # decode RLE, select subset of masks included in box, and cast to torch tensor
        g = torch.tensor(rle.decode(gm)[r1:r2, c1:c2], dtype=torch.bool).to(device)
        p = torch.tensor(rle.decode(pm)[r1:r2, c1:c2], dtype=torch.bool).to(device)
        g = torch.nn.ConstantPad2d(1,0)(g)
        p = torch.nn.ConstantPad2d(1,0)(p)

        B_g = g^binary_erosion(g)
        B_p = p^binary_erosion(p)

        # indices of pixels included in ground truth and predicted masks
        gt_where = torch.stack(torch.where(B_g), axis=1)
        pred_where = torch.stack(torch.where(B_p), axis=1)

        # compile ground truths without current gt mask
        t_drop = np.delete(t,m[0],0)

        # collaps roi on ground truths
        t_roi = torch.tensor(np.logical_or.reduce(t_drop,axis=0)[r1:r2,c1:c2],dtype=torch.bool).to(device)
        t_roi = torch.nn.ConstantPad2d(1,0)(t_roi)

        # indices of boundary particles involved in an overlap
        gt_where_o = torch.where(g[gt_where[:,0],gt_where[:,1]] & t_roi[gt_where[:,0],gt_where[:,1]])[0]
        pred_where_o = torch.where(p[pred_where[:,0],pred_where[:,1]] & t_roi[pred_where[:,0],pred_where[:,1]])[0]

        # distances from one contour to the other
        g_dist = _min_euclid(gt_where,pred_where)
        p_dist = _min_euclid(pred_where,gt_where)

        # parse overlapping particles from non-overlapping
        g_dist_o = g_dist[gt_where_o]
        g_dist_no = np.delete(g_dist,gt_where_o)

        p_dist_o = p_dist[pred_where_o]
        p_dist_no = np.delete(p_dist,pred_where_o)


        # calculate scores
        ## overlapping
        if p_dist_o.numel() and g_dist_o.numel():
            Pc_o = 1/p_dist_o.shape[0]*(p_dist_o[p_dist_o<theta].shape[0])
            Rc_o = 1/g_dist_o.shape[0]*(g_dist_o[g_dist_o<theta].shape[0])
            
            if Pc_o+Rc_o ==0:
                BF_scores_o.append(0)
            else:
                F1_o = 2*Pc_o*Rc_o/(Pc_o+Rc_o)
                BF_scores_o.append(F1_o)
            
        elif p_dist_o.numel() or g_dist_o.numel():
            BF_scores_o.append(0)
        else:
            pass

        ## non-overlapping
        if p_dist_no.numel() and g_dist_no.numel():
            Pc_no = 1/p_dist_no.shape[0]*(p_dist_no[p_dist_no<theta].shape[0])
            Rc_no = 1/g_dist_no.shape[0]*(g_dist_no[g_dist_no<theta].shape[0])
            
            if Pc_no+Rc_no ==0:
                BF_scores_no.append(0)
            else:
                F1_no = 2*Pc_no*Rc_no/(Pc_no+Rc_no)
                BF_scores_no.append(F1_no)

        elif p_dist_no.numel() or g_dist_no.numel():
            BF_scores_no.append(0)
        else:
            pass

    BF_mean_o = np.mean(BF_scores_o)
    BF_mean_no = np.mean(BF_scores_no)
    
    return BF_scores_o, BF_scores_no









