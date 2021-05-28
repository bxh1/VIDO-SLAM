#!/usr/bin/python
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import os
sys.path.append("../"+os.path.dirname(__file__))
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils import cv2_util
import ros_numpy

import numpy as np
import math
import matplotlib.path
import torch

from predictor import COCODemo

from mask_rcnn.srv import MaskRcnn, MaskRcnnResponse
from sensor_msgs.msg import Image
import struct
import rospy
import random


import time

class MaskRcnnRos():#RosCppCommunicator):

    def __init__(self, config_path =os.path.dirname(__file__) + "/configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"):
        self.model_config_path = config_path
        cfg.merge_from_file(self.model_config_path)
        cfg.merge_from_list([])
        cfg.freeze()

        self._greyscale_palette = (2 * 25 - 1)


        # prepare object that handles inference plus adds predictions on top of image
        self.coco_demo = COCODemo(
            cfg,
            confidence_threshold=0.8,
            show_mask_heatmaps=False,
            masks_per_dim=1,
            min_image_size=800
        )

        self._greyscale_colours = self._generate_grayscale_values()



        self.mask_rcnn_service = rospy.Service("MaskRcnnService",MaskRcnn, self.mask_rcnn_service_callback)
        print("Ready to receive mask rcnn service calls.")
        self.mask_rcnn_publisher = rospy.Publisher('mask_rcnn_img', Image, queue_size=10)


    @torch.no_grad()
    def mask_rcnn_service_callback(self, req):
        response = MaskRcnnResponse()
        input_image = ros_numpy.numpify(req.input_image)
        response_image, labels, label_indexs = self.analyse_image(input_image)
        display_image = response_image * 48
        output_image_msg = ros_numpy.msgify(Image, response_image, encoding='mono8')
        display_image_msg = ros_numpy.msgify(Image, display_image, encoding='mono8')
        self.mask_rcnn_publisher.publish(display_image_msg)
        response.success = True
        response.output_mask = output_image_msg
        response.labels = labels
        response.label_indexs = label_indexs

        del response_image
        del labels
        del label_indexs
        return response

    @torch.no_grad()
    def analyse_image(self, image):
        predictions = self.coco_demo.compute_prediction(image)
        top_predictions = self.coco_demo.select_top_predictions(predictions)
        return self.create_pixel_masks(image, top_predictions)
       

    def create_pixel_masks(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        width = image.shape[0]
        height = image.shape[1]
        blank_mask = np.zeros((width, height),np.uint8)

        if predictions is None:
            return blank_mask, [], []
        masks = predictions.get_field("mask")
        label_indexs = predictions.get_field("labels").numpy()
        labels = self.convert_label_index_to_string(label_indexs)


        # colours = self.get_greyscale_colours(label_indexs)
        
        if masks.ndim < 3:
            masks = np.expand_dims(masks, axis=0)
            masks = np.expand_dims(masks, axis=0)

        #TODO: make sure there is a boarder around each mask so that they are definetely considered
        #separate objects
        for mask, semantic_index in zip(masks, label_indexs):
            thresh = mask[0, :, :].astype(np.uint8) * semantic_index
            # print(mask.shape)
            # thresh = mask.astype(np.uint8) * colour
            # print(thresh)
        
            blank_mask += thresh
            # contours, hierarchy = cv2_util.findContours(
            #     thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            # )
            # # contours = contours[0] if len(contours) == 2 else contours[1]
            # image = cv2.drawContours(image, contours, -1, color, 3)

        composite = blank_mask

        return composite, labels, label_indexs

    def convert_label_index_to_string(self, labels):
        return [self.coco_demo.CATEGORIES[i] for i in labels]

    def get_single_label_from_index(self, label):
        return self.coco_demo.CATEGORIES[label]

    def get_greyscale_colours(self, label_index):
        return self._greyscale_colours[label_index]
        

    def _generate_grayscale_values(self):
        """[Generates n number of distinct values between 1 and 255 for each label. This should be 
        used for visualisation purposes only as VDOSLAM just needs a distinct value]

        Returns:
            [List]: [List of values]
        """
        numer_of_cats = len(self.coco_demo.CATEGORIES)  
        categories_index = np.linspace(0, numer_of_cats, numer_of_cats + 1)
        colors = np.array(categories_index) * self._greyscale_palette
        colors = (colors % 255).astype("uint8")
        print(type(colors))
        return colors



if __name__ == "__main__":
    rospy.init_node('mask_rcnn')
    mask_rcnn_ros = MaskRcnnRos()
    rospy.spin()

