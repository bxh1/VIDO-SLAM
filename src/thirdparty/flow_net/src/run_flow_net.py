#!/usr/bin/python
import time
import os
import struct
import rospy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
import sys
import ros_numpy
import math

from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms

from imageio import imread, imwrite



from layers import Network
from flow_net.srv import FlowNet, FlowNetResponse
sys.path.append("../"+os.path.dirname(__file__))

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2


class FlowNetRos():
    def __init__(self, model_path= os.path.dirname(__file__) +"/models/network-default.pytorch"):
        #RosCppCommunicator.__init__(self)
       
        self.model_path = model_path
       
        torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
        cudnn.enabled = True # make sure to use cudnn for computational performance

        self.network = Network(self.model_path).cuda().eval()

        #set up service calls
        self.flow_net_service = rospy.Service("FlowNetService",FlowNet, self.flow_net_service_callback)
        print("Ready to receive flownet service calls.")
        self.flownet_publisher = rospy.Publisher('flow_img', Image, queue_size=10)
    

    @torch.no_grad()
    def flow_net_service_callback(self, req):
        
        previous_image = ros_numpy.numpify(req.previous_image)
        current_image = ros_numpy.numpify(req.current_image)
        output_image = self.analyse_flow(previous_image, current_image)
        rgb_flow = self.flow2rgb(output_image)
        flow_image_msg = ros_numpy.msgify(Image, rgb_flow, encoding='rgb8')
        self.flownet_publisher.publish(flow_image_msg)

        output_image_msg = ros_numpy.msgify(Image, output_image, encoding='32FC2')
        response = FlowNetResponse()
        response.success = True
        response.output_image = output_image_msg

        return response

    @torch.no_grad()
    def analyse_flow(self, previous_image, current_image):

        #convert to tensor array
        tenFirst = torch.FloatTensor(np.ascontiguousarray(np.array(previous_image)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
        tenSecond = torch.FloatTensor(np.ascontiguousarray(np.array(current_image)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
        assert(tenFirst.shape[1] == tenSecond.shape[1])
        assert(tenFirst.shape[2] == tenSecond.shape[2])

        intWidth = tenFirst.shape[2]
        intHeight = tenFirst.shape[1]

        # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
        # assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

        tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
        tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

        tenFlow = torch.nn.functional.interpolate(input=self.network(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        flow = tenFlow[0, :, :, :].cpu()
        flow = flow.squeeze(0)

        #this is a 2 x R X C vector -> we need it to be a R X C X 2
        flow_map_np = flow.detach().cpu().numpy()
        flow_map_np = np.moveaxis(flow_map_np, 0, -1)
        flow_map_np = flow_map_np.astype(np.float32)


        del tenFlow
        del tenPreprocessedFirst
        del tenPreprocessedSecond
        del tenFirst
        del tenSecond
        
        return flow_map_np

    def flow2rgb(self, flow_map_np, normalize = True):
        hsv = np.zeros((flow_map_np.shape[0], flow_map_np.shape[1], 3), dtype=np.uint8)
        flow_magnitude, flow_angle = cv2.cartToPolar(flow_map_np[..., 0].astype(np.float32), flow_map_np[..., 1].astype(np.float32))

        # A couple times, we've gotten NaNs out of the above...
        nans = np.isnan(flow_magnitude)
        if np.any(nans):
            nans = np.where(nans)
            flow_magnitude[nans] = 0.

        # Normalize
        hsv[..., 0] = flow_angle * 180 / np.pi / 2
        if normalize is True:
            hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 1] = flow_magnitude
        hsv[..., 2] = 255
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img


if __name__ == '__main__':
    rospy.init_node('flow_net')
    flow_net_ros = FlowNetRos()
    rospy.spin()




