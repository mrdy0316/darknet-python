from ctypes import *
import math
import random
import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import os, os.path

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

def get_color(name):
    color_list = ((1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1))
    if name in get_color.name_list:
        return np.array(color_list[get_color.name_list.index(name) % 6])
    else:
        get_color.name_list.append(name)
        return np.array(color_list[(len(get_color.name_list)-1) % 6])
get_color.name_list = []

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

thresh=.5
hier_thresh=.5
nms=.45
net = load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
meta = load_meta(b"./cfg/coco.data")

# Real Sense Function
# Configure depth and color streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_record_to_file('object_detection.bag')

# Start streaming
pipeline = rs.pipeline()
pipeline.start(config)

cur_time = cv2.getTickCount()
plt.figure()
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame: 
            continue 

        # Intrinsics & Extrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # Convert images to numpy arrays 
        depth_image = np.asanyarray(depth_frame.get_data()) 
        color_image = np.asanyarray(color_frame.get_data())

        # Detection
        im = array_to_image(color_image)
        num = c_int(0)
        pnum = pointer(num)
        predict_image(net, im)
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): do_nms_obj(dets, num, meta.classes, nms);

        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    color = get_color(meta.names[i])
                    b = dets[j].bbox
                    depth = depth_frame.get_distance(int(b.x), int(b.y))
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin,[int(b.x), int(b.y)],depth)
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h), (depth_point[0], depth_point[1], depth_point[2])))
                    cv2.rectangle(color_image,(int(b.x-b.w/2),int(b.y-b.h/2)),(int(b.x+b.w/2),int(b.y+b.h/2)),(int(255*color[0]), int(255*color[1]), int(255*color[2])),3)

                    # plot
                    plt.scatter(depth_point[0], depth_point[2],c=tuple(color),alpha=0.7, s=100)

        res = sorted(res, key=lambda x: -x[1])
        # free_image(im)
        # free_detections(dets, num)
        # print(res)
        # print time
        new_time = cv2.getTickCount()
        print((new_time - cur_time) / cv2.getTickFrequency())
        cur_time = new_time
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first) 
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) 
        
        # Stack both images horizontally 
        images = np.hstack((color_image, depth_colormap)) 
        
        # Show images 
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE) 
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        plt.pause(0.1)

finally:
    # Stop streaming 
    pipeline.stop()

# csvdir = "../realsense-data/csv/"
# csvprefix = "_Depth_"
# pngdir = "../realsense-data/png/"
# pngprefix = "_Color_"

# detectedpngdir = "../realsense-data/detected_png/"
# detectedpltdir = "../realsense-data/detected_plt/"

# f_x = 385.522
# c_x = 318.766
# DEPTH_SCALE = 1000.0

# xmax = 3000
# xmin = -3000
# def array_to_image(arr):
#     arr = arr.transpose(2,0,1)
#     c = arr.shape[0]
#     h = arr.shape[1]
#     w = arr.shape[2]
#     arr = (arr/255.0).flatten()
#     data = c_array(c_float, arr)
#     im = IMAGE(w,h,c,data)
#     return im

# def get_color(name):
#     color_list = ((1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1))
#     if name in get_color.name_list:
#         return np.array(color_list[get_color.name_list.index(name) % 6])
#     else:
#         get_color.name_list.append(name)
#         return np.array(color_list[(len(get_color.name_list)-1) % 6])
# get_color.name_list = []
# def array_to_image(arr):
#     arr = arr.transpose(2,0,1)
#     c = arr.shape[0]
#     h = arr.shape[1]
#     w = arr.shape[2]
#     arr = (arr/255.0).flatten()
#     data = c_array(c_float, arr)
#     im = IMAGE(w,h,c,data)
#     return im

# def get_color(name):
#     color_list = ((1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1))
#     if name in get_color.name_list:
#         return np.array(color_list[get_color.name_list.index(name) % 6])
#     else:
#         get_color.name_list.append(name)
#         return np.array(color_list[(len(get_color.name_list)-1) % 6])
# get_color.name_list = []
# def array_to_image(arr):
#     arr = arr.transpose(2,0,1)
#     c = arr.shape[0]
#     h = arr.shape[1]
#     w = arr.shape[2]
#     arr = (arr/255.0).flatten()
#     data = c_array(c_float, arr)
#     im = IMAGE(w,h,c,data)
#     return im

# def get_color(name):
#     color_list = ((1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1))
#     if name in get_color.name_list:
#         return np.array(color_list[get_color.name_list.index(name) % 6])
#     else:
#         get_color.name_list.append(name)
#         return np.array(color_list[(len(get_color.name_list)-1) % 6])
# get_color.name_list = []
# def array_to_image(arr):
#     arr = arr.transpose(2,0,1)
#     c = arr.shape[0]
#     h = arr.shape[1]
#     w = arr.shape[2]
#     arr = (arr/255.0).flatten()
#     data = c_array(c_float, arr)
#     im = IMAGE(w,h,c,data)
#     return im

# def get_color(name):
#     color_list = ((1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1))
#     if name in get_color.name_list:
#         return np.array(color_list[get_color.name_list.index(name) % 6])
#     else:
#         get_color.name_list.append(name)
#         return np.array(color_list[(len(get_color.name_list)-1) % 6])
# get_color.name_list = []
# if __name__ == "__main__":
#     net = load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
#     meta = load_meta("cfg/coco.data")
#     framenum = len([name for name in os.listdir(pngdir) if os.path.isfile(os.path.join(pngdir, name))])
#     for i in range(1,framenum):
#         r = detect3d(net, meta, i)
# print(i,": ",r)