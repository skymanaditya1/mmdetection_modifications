from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import cv2
import matplotlib.pyplot as plt

import os.path as osp
import os
import random
from glob import glob
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

# load the checkpoint and initialize the detector
config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
detection_model = init_detector(config, checkpoint, device='cuda:0')

# Image transformations
scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def generate_object_segmentation(scene_image, save_scene_dir):
    result = inference_detector(detection_model, scene_image)
    # Save the generated segments to disk
    out_file = osp.join(save_scene_dir, 'generated_' + osp.basename(scene_image))
    show_result_pyplot(detection_model, scene_image, result, out_file=out_file, score_thr=0.3)

    return result

# method for computing the intersection over union (IOU) for two bounding boxes
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # Compute the IOU
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# Use the full object arrays and masks to generate the object segmentations 
# Extract the object bounding boxes and the corresponding masks 
# For two bounding boxes, check if the iou between them is greater than threshold, reject bb with lower score
# Generate the mask over the remaining bounding boxes
# Use the generated image segments for performing feature matching with the target mesh objects 
# This can be used for grasp verification as well as for object matching (replacing SuperGlue)
# Generate features using multiple networks (model ensembling) and combine their features 
def get_object_segments(result):
    not_none = 0
    object_arrays = list()
    object_masks = list()
    for i, val in enumerate(result[0]):
        if len(val) != 0:
            not_none += 1
            object_arrays.append(val)
            object_masks.append(result[1][i])
        # print(i, val)
    print(f'Arrays that don\'t have zero length are : {not_none}')

    objects = list()
    masks = list()
    for i, object_array in enumerate(object_arrays):
        # print(i, object_array)
        objects.append(object_array[0])
        masks.append(object_masks[i][0])

    return objects, masks

def purge_bounding_boxes(objects, masks):        
    # Compute the iou between two bounding boxes and remove the one with the lower confidence 
    for bb1_index in range(len(objects)):
        for bb2_index in range(bb1_index+1, len(objects)):
            if objects[bb1_index] is None or objects[bb2_index] is None:
                continue

            # Get the coordinate of the bounding boxes
            bb1 = {'x1': int(objects[bb1_index][0]), 
            'x2': int(objects[bb1_index][2]), 
            'y1': int(objects[bb1_index][1]), 
            'y2': int(objects[bb1_index][3])}

            bb2 = {'x1': int(objects[bb2_index][0]), 
            'x2': int(objects[bb2_index][2]), 
            'y1': int(objects[bb2_index][1]), 
            'y2': int(objects[bb2_index][3])}
            
            iou = get_iou(bb1, bb2)
            if iou>=0.5:
                # set the bb with the lower confidence to None
                bounding_box1 = objects[bb1_index]
                bounding_box2 = objects[bb2_index]
                # print(f'The indices are : {bb1_index}, {bb2_index}')
                
                if bounding_box1[4] < bounding_box2[4]:
                    objects[bb1_index] = None
                    masks[bb1_index] = None
                else:
                    objects[bb2_index] = None
                    masks[bb2_index] = None

    # Purge all bbs that are None
    updated_bbs = [bb for bb in objects if bb is not None]
    updated_masks = [mask for mask in masks if mask is not None]
    # print(len(updated_bbs), len(updated_masks))
    # print(updated_masks)

    return updated_bbs, updated_masks

# function to generate a 5 digit random alphanumeric number
def get_random(CHAR_LEN=5):
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    random_chars = list()
    for i in range(CHAR_LEN):
        random_chars.append(chars[random.randint(0, len(chars)-1)])

    random_filename = ''.join(random_chars)
    return random_filename

def segment_object_from_mask(scene_image, updated_bbs, updated_masks, segments_save_dir_name):
    # segmented_images = list()
    for i, sample_bb in enumerate(updated_bbs):
        image = cv2.cvtColor(cv2.imread(scene_image), cv2.COLOR_BGR2RGB)
        object_mask = ~updated_masks[i]
        image[object_mask] = 0
        cropped_image = image[int(sample_bb[1]):int(sample_bb[3]), int(sample_bb[0]):int(sample_bb[2])]
        
        # plt.figure()
        # plt.imshow(cropped_image)
        
        # save the segmented image
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        # segmented_images.append(cropped_image)
        save_filepath = os.path.join(segments_save_dir_name, str(get_random()) + '.jpg')
        cv2.imwrite(save_filepath, cropped_image)

        # return segmented_images

# Build an ensemble network which consists of multiple pretrained feature extractor networks 

# generate the embedding from the vgg16 network 
def vgg16_embedding(image_name, ):
    # Load the pretrained model
    model = models.vgg16(pretrained=True)
    # Use the model object to select the desired layer
    layer = model._modules.get('avgpool')

    # Set model to evaluation mode
    model.eval()
    
    dimension = 25088

    return get_vector(image_name, dimension, layer, model)
    

# generate the embedding from the resnet network 
def resnet18_embedding(image_name):
    # Load the pretrained resnet18 model 
    model = models.resnet18(pretrained=True)
    # Use the model object to select the desired layer 
    layer = model._modules.get('avgpool')
    
    # set model to evaluation mode 
    model.eval()
    
    dimension = 512
    
    return get_vector(image_name, dimension, layer, model)
    

# generate the embedding from the Alexnet network 
def alexnet_embedding(image_name):
    # Load the pretrained faster rcnn model 
    model = models.alexnet(pretrained=True)
    # Use the model object to select the desired layer 
    layer = model._modules.get('avgpool')
    
    # set model to evaluation mode 
    model.eval() 
    
    dimension = 9216
    
    return get_vector(image_name, dimension, layer, model)


# generate the embedding from the Densenet network 
def densenet_embedding(image_name):
    # Load the pretrained faster rcnn model 
    model = models.densenet161(pretrained=True)
    # Use the model object to select the desired layer 
    layer = model._modules.get('features')
    
    # set model to evaluation mode 
    model.eval() 
    
    dimension = 108192
    
    return get_vector(image_name, dimension, layer, model)


# generate the embedding from the Inceptionnet network
def inceptionnet_embedding(image_name):
    # Load the pretrained faster rcnn model 
    model = models.inception_v3(pretrained=True)
    # Use the model object to select the desired layer 
    layer = model._modules.get('avgpool')
    
    # set model to evaluation mode 
    model.eval() 
    
    dimension = 2048
    
    return get_vector(image_name, dimension, layer, model)


def get_vector(image_name, dimension, layer, model):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))[:3]).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(dimension)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.view(-1))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


# Find the cosine similarity between two image vectors
def cosine_similarity(image1, image2, func):
    pic_one_vector = func(image1)
    pic_two_vector = func(image2)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(pic_one_vector.unsqueeze(0),
                  pic_two_vector.unsqueeze(0))
    # print('\nCosine similarity: {0}\n'.format(cos_sim))
    return cos_sim

# Given segmented images in the scenes, compare the cosine similarity between the segmented images and the target 
def voting_function(segment_images, target_image):
    # There are two voting functions that can be used
    # Assign score to the highest-ranked object using a function 
    # Aggregate ranks against each index, and use index with the lowest rank

    match_indices = list()
    feature_functions = [vgg16_embedding, resnet18_embedding, alexnet_embedding, densenet_embedding, inceptionnet_embedding]

    # segmented_dir = 'saved_segments/new_segments'
    # segment_images = glob(segmented_dir + '/*.jpg')
    # target_image = os.path.join('saved_segments/matches', 'orange.png')

    majority_match = np.zeros(len(segment_images))
    full_scores = np.zeros(len(segment_images))

    for feature_function in feature_functions:
        print(f'Using feature function : {feature_function}')
        cosine_scores = np.array([])
        for index, img in enumerate(segment_images):
            cosine_sim = cosine_similarity(img, target_image, feature_function)
            cosine_scores = np.append(cosine_scores, cosine_sim.item())
        segment_match_index = np.argmax(cosine_scores)
        sorted_ranks = np.argsort(cosine_scores)
        sorted_ranks = np.argsort(sorted_ranks)
        full_scores += sorted_ranks
        majority_match[segment_match_index] += 1

    print(majority_match)
    print(full_scores)
    print(segment_images)

    majority_vote_index = np.argmax(majority_match)
    full_vote_index = np.argmax(full_scores)

    print(f'Highest using majority vote : {os.path.basename(segment_images[majority_vote_index])}')
    print(f'Highest using full vote : {os.path.basename(segment_images[full_vote_index])}')
    print(f'Ground truth image : {os.path.basename(target_image)}')

if __name__ == '__main__':
    saved_scenes_dir = 'saved_scenes'
    os.makedirs(saved_scenes_dir, exist_ok=True)
    saved_segments_dir = 'saved_segments'
    current_segments_dir = osp.join(saved_segments_dir, get_random()) # save all generated segments

    os.makedirs(saved_segments_dir, exist_ok=True)
    os.makedirs(current_segments_dir, exist_ok=True)

    scene_image = 'saved_segments/2-1-1.png'
    target_image = 'saved_segments/155mx.jpg'

    result = generate_object_segmentation(scene_image, saved_scenes_dir)
    objects, masks = get_object_segments(result)
    bbs, masks = purge_bounding_boxes(objects, masks)
    segment_object_from_mask(scene_image, bbs, masks, current_segments_dir)

    segment_images = glob(current_segments_dir + '/*.jpg')

    voting_function(segment_images, target_image)