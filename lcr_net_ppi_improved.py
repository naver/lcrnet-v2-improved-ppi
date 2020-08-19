""" LCR-Net: Localization-Classification-Regression for Human Pose
Copyright (C) 2020 NAVER Corp.
​
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
​
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
​
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>"""

import numpy as np

"""
2D boxes are N*4 numpy arrays (float32) with each row at format <xmin> <ymin> <xmax> <ymax>
"""

def area2d(b):
    """ compute the areas for a set of 2D boxes"""
    return (b[:,2]-b[:,0]+1)*(b[:,3]-b[:,1]+1)

def overlap2d(b1, b2):
    """ compute the overlaps between a set of boxes b1 and 1 box b2 """
    xmin = np.maximum( b1[:,0], b2[:,0] )
    xmax = np.minimum( b1[:,2]+1, b2[:,2]+1)
    width = np.maximum(0, xmax-xmin)
    ymin = np.maximum( b1[:,1], b2[:,1] )
    ymax = np.minimum( b1[:,3]+1, b2[:,3]+1)
    height = np.maximum(0, ymax-ymin)   
    return width*height          

def iou2d(b1, b2):
    """ compute the IoU between a set of boxes b1 and 1 box b2"""
    if b1.ndim == 1: b1 = b1[None,:]
    if b2.ndim == 1: b2 = b2[None,:]
    assert b2.shape[0]==1
    o = overlap2d(b1, b2)
    return o / ( area2d(b1) + area2d(b2) - o ) 



def convert_to_ppi_format(scores, boxes, pose2d, pose3d, score_th=None, tonumpy=True):
  if score_th is None: score_th = 0.1/(scores.shape[1]-1)
  boxindices, idxlabels = np.where(scores[:,1:]>=score_th) # take the Ndet values with scores over a threshold
  s_scores = scores[boxindices, 1+idxlabels] # Ndet (number of estimations with score over score_th)
  s_boxes = boxes[boxindices,:]  # Ndetx4 (does not depend on label)
  s_pose2d = pose2d[boxindices, idxlabels, : ,:] # NdetxJx2
  s_pose3d = pose3d[boxindices, idxlabels, : ,:] # NdetxJx3
  
  return s_scores, s_boxes, s_pose2d, s_pose3d, boxindices, idxlabels 

indTorsoHead = {13: [4,5,10,11,12], 14: [4,5,10,11,12,13]}
indLower     = {13: range(0,4), 14: range(0,4)}  # indices of joints for lower body
indUpper     = {13: range(4,13), 14: range(4,14)}  # indices of joints for upper body

def LCRNet_PPI_improved(scores, boxes, pose2d, pose3d, resolution, K=None, score_th=None, th_pose3D=0.3, th_iou=0.2, iou_func='bbox', min_mode_score=0.05, th_persondetect=0.002, verbose=False):
    """
    this function extends the Pose Proposals Integration (PPI) from LCR-Net in order to also handle hands (J=21), faces (J=84) and bodies with 13 or 14 joints.    
    
    scores, boxes, pose2d, pose3d are numpy arrays of size [Nboxes x (Nclasses+1)], [Nboxes x 4], [Nboxes x Nclasses x J x 2] and [Nboxes x Nclasses x J x 3] resp.
   
    resolution: tuple (height,width) containing image resolution
    K: number of classes, without considering lower/upper (K=10)
    score_th: only apply ppi on pose proposals with score>=score_th (None => 0.1/K)
    th_pose3D: threshold for groupind pose in a mode, based on mean dist3D of the joints
    th_iou: 2D overlap threshold to group the pose proposals
    iou_func: how 2d overlap is defined 
    min_mode_score: we consider modes with at least this score (save time and do not affect 1st mode)
    th_persondetect: we remove final detection whose cumscore is below this threshold

    return a list of detection with the following fields:
      * score: cumulative score of the detection
      * pose2d: Jx2 numpy array
      * pose3d: Jx3 numpy array
    """
  
    if K is None: K = scores.shape[1]
    s_scores, s_boxes, s_pose2d, s_pose3d, boxindices, idxlabels = convert_to_ppi_format(scores, boxes, pose2d, pose3d, score_th=score_th)
  
    H,W = resolution
    wh = np.array([[W,H]], dtype=np.float32)
    J = s_pose2d.shape[1]
    if not J in [13,14]: assert iou_func in ['bbox', 'mje']

    # compute bounding boxes from 2D poses truncated by images boundaries
    if iou_func=='bbox_torsohead': # using torso+head keypoints only    
        xymin = np.minimum( wh-1, np.maximum(0, np.min(s_pose2d[:,indTorsoHead[J],:], axis=1)))
        xymax = np.minimum( wh-1, np.maximum(0, np.max(s_pose2d[:,indTorsoHead[J],:], axis=1)))
        bbox_headtorso = np.concatenate( (xymin,xymax), axis=1)
    else:  # using all keypoints
        xymin = np.minimum( wh-1, np.maximum(0, np.min(s_pose2d, axis=1)))
        xymax = np.minimum( wh-1, np.maximum(0, np.max(s_pose2d, axis=1)))
        bboxes = np.concatenate( (xymin,xymax), axis=1)
    # define iou metrics
    def compute_overlapping_poses(bboxes, poses2d, a_bbox, a_pose2d, th_iou):
        assert a_pose2d.ndim==2 and a_bbox.ndim==1
        a_bbox = a_bbox[None,:]
        a_pose2d = a_pose2d[None,:,:]
        assert bboxes.ndim==2 and poses2d.ndim==3
        if iou_func=='bbox' or iou_func=='bbox_torsohead':
            iou = iou2d(bboxes, a_bbox)
            return np.where( iou>th_iou )[0]
        elif iou_func=='torso' or iou_func=='torsoLR':
            indices = [4,5,10,11]
            lr_indices = [5,4,11,10]
        elif iou_func=='torsohead' or iou_func=='torsoheadLR':
            indices = [4,5,10,11,12] if J==13 else [4,5,10,11,12,13]
            lr_indices = [5,4,11,10,12]  if J==13 else [5,4,11,10,12,13]
        elif iou_func=='head':
            indices = [12]  if J==13 else [12,13]
        elif iou_func=='shoulderhead' or iou_func=='shoulderheadLR':
            indices = [10,11,12]  if J==13 else [10,11,12,13]
            lr_indices = [11,10,12]  if J==13 else [11,10,12,13]
        elif iou_func=='mje':
            indices = list(range(J))
        else:
            raise NotImplementedError('ppi.py: unknown iou_func')
        indices = np.array(indices, dtype=np.int32)
        if iou_func.endswith('LR'):
            lr_indices = np.array(lr_indices, dtype=np.int32)
            a = np.minimum( np.mean(np.sqrt( np.sum( (poses2d[:,   indices,:]-a_pose2d[:,indices,:])**2, axis=2)), axis=1),
                            np.mean(np.sqrt( np.sum( (poses2d[:,lr_indices,:]-a_pose2d[:,indices,:])**2, axis=2)), axis=1) )
        else:
            a = np.mean(np.sqrt(np.sum( (poses2d[:,indices,:]-a_pose2d[:,indices,:])**2, axis=2)), axis=1)
        b = 2*np.max( a_bbox[:,2:4]-a_bbox[:,0:2]+1 )
        return np.where( a/b < th_iou)[0]

  
    # group pose proposals according to 2D IoU
    Persons = [] # list containing the detected people, each person being a tuple ()
    remaining_pp = range( s_pose2d.shape[0] )
    while len(remaining_pp)>0:
        # take highest remaining score
        imax = np.argmax( s_scores[remaining_pp] )
        # consider the pose proposals with high 2d overlap
        this = compute_overlapping_poses( s_boxes[remaining_pp,:], s_pose2d[remaining_pp,:], s_boxes[remaining_pp[imax],:], s_pose2d[remaining_pp[imax],:,:], th_iou)
        this_pp = np.array(remaining_pp, dtype=np.int32)[this]
        # add the person and delete the corresponding pp
        Persons.append( (this_pp, np.sum(s_scores[this_pp])) )
        remaining_pp = [p for p in remaining_pp if not p in this_pp]
    if verbose: print("{:d} persons/groups of poses found".format(len(Persons)))


    Detected = []
    # find modes for each person
    for iperson, (pplist, cumscore) in enumerate(Persons):

        remaining_pp = list(pplist.copy()) # create a copy, list of pp that are not assigned to any mode
        Modes = []

        while len(remaining_pp)>0:

            # next anchor pose mode is defined as the top regscore among unassigned poses
            imax = np.argmax( s_scores[remaining_pp] )
            maxscore = s_scores[remaining_pp[imax]]
            if maxscore<min_mode_score and len(Modes)>0: break # stop if score not sufficiently high and already created a mode
            
            # select PP (from the entire set) close to the center of the mode
            mode_pose3D = s_pose3d[ remaining_pp[imax], :,:]
            #dist3D = np.mean( np.sqrt( (mode_pose3D[ 0:13]-regpose3d[pplist, 0:13])**2 + \
            #                           (mode_pose3D[13:26]-regpose3d[pplist,13:26])**2 + \
            #                           (mode_pose3D[26:39]-regpose3d[pplist,26:39])**2 ), axis=1)
            dist3D = np.mean(np.sqrt( np.sum( (mode_pose3D-s_pose3d[pplist,:,:])**2, axis=2)), axis=1)
            this = np.where( dist3D < th_pose3D )[0]
            
            # compute the output for this mode
            this_pp = pplist[this]            
            weights = s_scores[this_pp]            
            
            # upper body is average weights by the scores
            hand_isright = None
            if J in [13,14]:
                p3d = np.empty( (J,3), dtype=np.float32)
                p2d = np.empty( (J,2), dtype=np.float32)
                cumscore = np.sum(weights)
                p3d[ indUpper[J], :] = np.sum(weights[:,None,None] * s_pose3d[this_pp,:,:][:,indUpper[J],:], axis=0) / cumscore
                p2d[ indUpper[J], :] = np.sum(weights[:,None,None] * s_pose2d[this_pp,:,:][:,indUpper[J],:], axis=0) / cumscore
                
                assert idxlabels is not None
                # for lower body, we downweight upperbody scores
                this_ub = np.where(idxlabels[this_pp]> K)[0] # anchor pose for upper body
                weights[ this_ub ] *= 0.1
                cumscoreBot = np.sum(weights)
                p3d[ indLower[J], :] = np.sum(weights[:,None,None] * s_pose3d[this_pp,:,:][:,indLower[J],:], axis=0) / cumscoreBot
                p2d[ indLower[J], :] = np.sum(weights[:,None,None] * s_pose2d[this_pp,:,:][:,indLower[J],:], axis=0) / cumscoreBot
            else:
                cumscore = np.sum(weights)
                p3d = np.sum(weights[:,None,None] * s_pose3d[this_pp,:,:], axis=0) / cumscore
                p2d = np.sum(weights[:,None,None] * s_pose2d[this_pp,:,:], axis=0) / cumscore
                if J==21:
                    hand_isright = (idxlabels[imax] < K)
              
            this_mode = {'score': cumscore, 'pose3d': p3d, 'pose2d': p2d}
            if hand_isright is not None: this_mode['hand_isright'] = hand_isright
            Modes.append( this_mode ) 
            
            # remove pp from the list to process
            remaining_pp = [p for p in remaining_pp if not p in this_pp]
        if verbose: print("Person {:d}/{:d} has {:d} mode(s)".format(iperson+1, len(Persons), len(Modes)))
            
        # keep the main mode for each person, only if score is sufficient high
        modes_score = np.array([m['score'] for m in Modes])
        bestmode = np.argmax( modes_score )
        if modes_score[bestmode] > th_persondetect:
            Detected.append( Modes[bestmode] )
        else:
            if verbose: print("\tdeleting this person because of too low score")
    if verbose: print('{:d} person(s) detected'.format(len(Detected)))
    # sort detection according to score
    Detected.sort( key=lambda d: d['score'], reverse=True)
    return Detected
