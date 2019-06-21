from darknet import *
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import os, os.path

csvdir = "../realsense-data/csv/"
csvprefix = "_Depth_"
pngdir = "../realsense-data/png/"
pngprefix = "_Color_"

detectedpngdir = "../realsense-data/detected_png/"
detectedpltdir = "../realsense-data/detected_plt/"

f_x = 385.522
c_x = 318.766
DEPTH_SCALE = 1000.0

xmax = 3000
xmin = -3000
xspan = (xmax - xmin) * 0.01
zmax = 6000
zspan = zmax * 0.01

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

def detect3d(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    str_im = str(image)
    arr_im = cv2.imread(pngdir+pngprefix+str_im+".png")
    #im = load_image(pngdir+str_im+".png", 0, 0)
    im = array_to_image(arr_im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []

    try:
         with open(csvdir+csvprefix+str_im+".csv") as f:
            reader = csv.reader(f)
            l = [row for row in reader]
    except:
        print("cannot open the file")

    plt.figure()
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                try:
                    b = dets[j].bbox
                    depth = float(l[int(b.y)][int(b.x)])
                    color = get_color(meta.names[i])
                    # calculate gloabal map point
                    p_z = depth * DEPTH_SCALE
                    p_x = (b.x - c_x) * p_z / f_x
                    plt.scatter(p_x,p_z,c=tuple(color),s=100)
                    # plt.scatter(p_x,p_z,c=tuple(color),s=100,alpha=0.7)
                    plt.text(p_x+xspan, p_z+zspan, "{0}({1})".format(meta.names[i],round(dets[j].prob[i],2)),fontsize=10)
                    # draw rectangles
                    cv2.rectangle(arr_im,(int(b.x-b.w/2),int(b.y-b.h/2)),(int(b.x+b.w/2),int(b.y+b.h/2)),tuple(255*color),3)
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h),depth))
                except:
                    print("calculation failed")
                    continue
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)

    plt.xlim(xmin,xmax)
    plt.ylim(0,zmax)
    plt.xlabel('x axis [mm]')
    plt.ylabel('z axis [mm]')
    plt.grid()
    plt.savefig(detectedpltdir+str_im+".png")
    #plt.show()
    #plt.pause(0.1)
    arr_im = cv2.cvtColor(arr_im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(detectedpngdir+str_im+".png",arr_im)
    cv2.imshow("output",arr_im)
    cv2.waitKey(1)

    return res

if __name__ == "__main__":
    net = load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
    meta = load_meta("cfg/coco.data")
    framenum = len([name for name in os.listdir(pngdir) if os.path.isfile(os.path.join(pngdir, name))])
    for i in range(1,framenum):
        r = detect3d(net, meta, i)
        print(i,": ",r)
