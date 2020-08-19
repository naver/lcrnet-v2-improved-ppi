""" LCR-Net: Localization-Classification-Regression for Human Pose
Copyright (C) 2017 Gregory Rogez & Philippe Weinzaepfel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>"""

import numpy as np

# 2D boxes are N*4 numpy arrays (float32) with each row at format <xmin> <ymin> <xmax> <ymax>
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


def LCRNet_PPI(res, K, resolution, th_pose3D=0.3, th_iou=0.2, iou_func='bbox', min_mode_score=0.05, th_persondetect=0.002, verbose=False, J=13):
    """
    Given the result from the network, perform Pose Proposals Integration.
    
    arguments:
    res: output from the LCR-Net
    K: number of anchor poses
    resolution: tuple (height,width) containing image resolution
    th_pose3D: threshold for groupind pose in a mode, based on mean dist3D of the joints
    th_iou: 2D overlap threshold to group the pose proposals
    iou_func: how 2d overlap is defined 
    min_mode_score: we consider modes with at least this score (save time and do not affect 1st mode)
    th_persondetect: we remove final detection whose cumscore is below this threshold
    J: number of joints in each pose 
     
    return a list of detected persons: 
        each one is a dictionary with 
        - cumscore
        - pose2d (2J dim)
        - pose3d (3J dim) 
    """    
    regpose2d = res['regpose2d']
    regpose3d = res['regpose3d']
    regscore = res['regscore']
    regclass = res['regclass']
    
    indX   = np.arange( 0, J) # indices of x coordinates in the 2D pose
    indY   = np.arange(J, 2*J) # indices of y coordinates in the 2D pose
    indBot = range( 0,  4) # indices of joints for lower body
    indUp  = range( 4, J) # indices of joints for upper body
    ind3dUp = np.array([J*i+j for i in range(3) for j in indUp], dtype=np.int32) # indices of coordinates for upperbody in 3D pose
    ind2dUp = np.array([J*i+j for i in range(2) for j in indUp], dtype=np.int32)
    ind3dBot = np.array([J*i+j for i in range(3) for j in indBot], dtype=np.int32)
    ind2dBot = np.array([J*i+j for i in range(2) for j in indBot], dtype=np.int32)
    
    H, W = resolution
    
    # compute bounding boxes from 2D poses truncated by image boundaries
    if iou_func=='bbox_torsohead': # using torso+head keypoints only
      indTorsoHeadX = np.array([4,5,10,11,12], dtype=np.int32)
      indTorsoHeadY = indTorsoHeadX+J
      xmin = np.minimum( np.maximum(0, np.min(regpose2d[:,indTorsoHeadX], axis=1)), W-1)
      ymin = np.minimum( np.maximum(0, np.min(regpose2d[:,indTorsoHeadY], axis=1)), H-1)
      xmax = np.minimum( np.maximum(0, np.max(regpose2d[:,indTorsoHeadX], axis=1)), W-1)
      ymax = np.minimum( np.maximum(0, np.max(regpose2d[:,indTorsoHeadY], axis=1)), H-1)
      bbox_headtorso = np.concatenate( (xmin[:,None],ymin[:,None],xmax[:,None],ymax[:,None]), axis=1)
    else:  # using all keypoints
      xmin = np.minimum( np.maximum(0, np.min(regpose2d[:,indX], axis=1)), W-1)
      ymin = np.minimum( np.maximum(0, np.min(regpose2d[:,indY], axis=1)), H-1)
      xmax = np.minimum( np.maximum(0, np.max(regpose2d[:,indX], axis=1)), W-1)
      ymax = np.minimum( np.maximum(0, np.max(regpose2d[:,indY], axis=1)), H-1)      
    bbox = np.concatenate( (xmin[:,None],ymin[:,None],xmax[:,None],ymax[:,None]), axis=1)
    def compute_overlapping_poses(bboxes, poses, abbox, apose, th_iou):
      assert apose.ndim==1 and abbox.ndim==1
      abbox = abbox[None,:]
      apose = apose[None,:]
      assert bboxes.ndim==2 and poses.ndim==2
      if iou_func=='bbox' or iou_func=='bbox_torsohead':
        iou = iou2d(bboxes, abbox)
        return np.where( iou>th_iou )[0]
      elif iou_func=='torso' or iou_func=='torsoLR':
        indices = [4,5,10,11]
        lr_indices = [5,4,11,10]
      elif iou_func=='torsohead' or iou_func=='torsoheadLR':
        indices = [4,5,10,11,12]
        lr_indices = [5,4,11,10,12]
      elif iou_func=='head':
        indices = [12]
      elif iou_func=='shoulderhead' or iou_func=='shoulderheadLR':
        indices = [10,11,12]
        lr_indices = [11,10,12]
      else:
        raise NotImplementedError('ppi.py: unknown iou_func')
      indices = np.array(indices, dtype=np.int32)
      if iou_func.endswith('LR'):
        lr_indices = np.array(lr_indices, dtype=np.int32)
        a = np.minimum(   np.mean(np.sqrt( (poses[:,   indices]-apose[:,indices])**2 + (poses[:,J+indices]-apose[:,J+   indices])**2 ), axis=1) ,
                          np.mean(np.sqrt( (poses[:,lr_indices]-apose[:,indices])**2 + (poses[:,J+indices]-apose[:,J+lr_indices])**2 ), axis=1) )
      else:
        a = np.mean(np.sqrt( (poses[:,indices]-apose[:,indices])**2 + (poses[:,J+indices]-apose[:,J+indices])**2 ), axis=1)
      b = 2*np.max( abbox[:,2:4]-abbox[:,0:2]+1 )
      return np.where( a/b < th_iou)[0]
    
    # group pose proposals according to 2D IoU
    Persons = [] # list containing the detected people, each person being a tuple ()
    remaining_pp = range( regpose2d.shape[0] )
    while len(remaining_pp)>0:
        # take highest remaining score
        imax = np.argmax( regscore[remaining_pp] )
        # consider the pose proposals with high 2d overlap
        this = compute_overlapping_poses( bbox[remaining_pp,:], regpose2d[remaining_pp,:], bbox[remaining_pp[imax],:], regpose2d[remaining_pp[imax],:], th_iou)
        this_pp = np.array(remaining_pp, dtype=np.int32)[this]
        # add the person and delete the corresponding pp
        Persons.append( (this_pp, np.sum(regscore[this_pp])) )

        remaining_pp = [p for p in remaining_pp if not p in this_pp]
    if verbose: print("{:d} persons/groups of poses found".format(len(Persons)))
    
    Detected = []
    # find modes for each person
    for iperson, (pplist, cumscore) in enumerate(Persons):

        remaining_pp = list(pplist.copy()) # create a copy, list of pp that are not assigned to any mode
        Modes = []

        while len(remaining_pp)>0:
            #import pdb; pdb.set_trace()
            # next anchor pose mode is defined as the top regscore among unassigned poses
            imax = np.argmax( regscore[remaining_pp] )
            maxscore = regscore[remaining_pp[imax]]
            if maxscore<min_mode_score and len(Modes)>0: break # stop if score not sufficiently high and already created a mode
            
            # select PP (from the entire set) close to the center of the mode
            mode_pose3D = regpose3d[ remaining_pp[imax], :]
            #dist3D = np.mean( np.sqrt( (mode_pose3D[ 0:13]-regpose3d[pplist, 0:13])**2 + \
            #                           (mode_pose3D[13:26]-regpose3d[pplist,13:26])**2 + \
            #                           (mode_pose3D[26:39]-regpose3d[pplist,26:39])**2 ), axis=1)
            dist3D = np.mean(np.sqrt((mode_pose3D[0:J] - regpose3d[pplist, 0:J]) ** 2 + \
                                     (mode_pose3D[J:2*J] - regpose3d[pplist, J:2*J]) ** 2 + \
                                     (mode_pose3D[2*J:3*J] - regpose3d[pplist, 2*J:3*J]) ** 2), axis=1)
            this = np.where( dist3D < th_pose3D )[0]
            
            # compute the output for this mode
            this_pp = pplist[this]            
            weights = regscore[this_pp]            
            pose3d = np.empty( (3*J,), dtype=np.float32)
            pose2d = np.empty( (2*J,), dtype=np.float32)
            
            # upper body is average weights by the scores
            cumscore = np.sum(weights)
            pose3d[ ind3dUp ] = np.sum( weights * regpose3d[this_pp,:][:,ind3dUp], axis=0 ) / cumscore
            pose2d[ ind2dUp ] = np.sum( weights * regpose2d[this_pp,:][:,ind2dUp], axis=0 ) / cumscore
            
            
            # for lower body, we downweight upperbody scores
            this_ub = np.where(regclass[this_pp]> K)[0] # anchor pose for upper body
            weights[ this_ub ] *= 0.1
            cumscoreBot = np.sum(weights)
            pose3d[ ind3dBot ] = np.sum( weights * regpose3d[this_pp,:][:,ind3dBot], axis=0 ) / cumscoreBot
            pose2d[ ind2dBot ] = np.sum( weights * regpose2d[this_pp,:][:,ind2dBot], axis=0 ) / cumscoreBot
            
            
            Modes.append( {'cumscore': cumscore, 'pose3d': pose3d, 'pose2d': pose2d} )
            
            # remove pp from the list to process
            remaining_pp = [p for p in remaining_pp if not p in this_pp]
        if verbose: print("Person {:d}/{:d} has {:d} mode(s)".format(iperson+1, len(Persons), len(Modes)))
        
        # keep the main mode for each person, only if score is sufficient high
        modes_score = np.array([m['cumscore'] for m in Modes])
        bestmode = np.argmax( modes_score )
        if modes_score[bestmode] > th_persondetect:
            Detected.append( Modes[bestmode] )
        else:
            if verbose: print("\tdeleting this person because of too low score")
    if verbose: print('{:d} person(s) detected'.format(len(Detected)))
    # sort detection according to score
    Detected.sort( key=lambda d: d['cumscore'], reverse=True)
    return Detected
                
                                            
        
    
    
    
