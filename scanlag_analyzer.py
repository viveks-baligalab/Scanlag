#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""@Author: Vivek Srinivas - Baliga Lab, ISB
- This is a program to identify hetergeneity in growth.
- Requires Images folder with with time lapse images aquired on a flatbed documents scanner.
##refer to examples files for format##
    Step 1: crop individual experiments
            crop_to_experiments(image_folder, out_dir = '__', ROI_dict = {dict with Label:(ROI)}, ROI_type = '__')
"""

import cv2
import numpy as np
import os
from imutils import contours
from skimage import measure
#import numpy as np
#import argparse
import imutils
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import optimize
#from scipy.stats import norm

def crop_circles(image):
    img = cv2.imread(image,0)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,350,500,minRadius=700,maxRadius=710)
    circles = np.uint16(np.around(circles))
    for n,i in enumerate(circles[0,:]):
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),10)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),20)
        # name the cirles
        cv2.putText(cimg,str(n),(i[0],i[1]),cv2.FONT_HERSHEY_PLAIN,8,(0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite("Circles.png",cimg)
    return img

def crop_image(image_file,folder,circles,names,time):
    img = cv2.imread(image_file,0)
    img = cv2.medianBlur(img,5)
    cropped = []
    for c in circles:
        #y,x = c[0],c[1]
        #cropped.append(img[y:y+2000,x:x+2000])
        x,y,d = c[0]-c[2],c[1]-c[2],c[2]*2
        cropped.append(img[y:y+d,x:x+d])
    for n,cro in enumerate(cropped):
        cv2.imwrite("%s/%s_%s.png"%(folder,names[n],time),cro)

def crop_images_in_folder(folder,out_folder, circles,names,to):
    for i in os.listdir(folder):
        try:
            t = int(i.split("_")[1].split(".")[0])
            crop_image("%s/%s"%(folder,i),out_folder,circles,names,int((t-to)/60.0))
        except:
            pass

def binarize_image(image):
    image = cv2.imread(image,0)
    thresh = cv2.threshold(image,0,255,cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh,None, iterations = 2)
    thresh = cv2.dilate(thresh, None, iterations = 4)
    #imS = cv2.resize(thresh, (int(thresh.shape[0]/2.),int(thresh.shape[1]/2.)))
    #cv2.imshow("Threshold",imS)#imwrite("Test/Threshold_image.png",thresh)
    #cv2.waitKey(0)
    return thresh

def mask_image(image,size):
    labels = measure.label(image,neighbors=8,background=0)
    mask = np.zeros(image.shape,dtype="uint8")
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(image.shape,dtype="uint8")
        labelMask[labels == label]=255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels<size:
            mask = cv2.add(mask,labelMask)
    #cv2.imwrite("Test/Masked_image.png",mask)
    return mask
        

def find_CFUs(image,image2,cell_type, folder):
    initial_image = cv2.imread(image2,0)
    cnts = cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = contours.sort_contours(cnts)[0]
    crop_rect = OrderedDict()
    spot_loc = OrderedDict()
    for (i,c) in enumerate(cnts):
        (x,y,w,h) = cv2.boundingRect(c)
        ((cX,cY),radius) = cv2.minEnclosingCircle(c)
        cv2.circle(initial_image,(int(cX),int(cY)),int(radius),
                   (0,255,0),1)
        spot_loc["%s_%s"%(cell_type,i)] = "%s_%s_%s"%(int(cX),int(cY),int(radius))
        crop_rect["%s_%s"%(cell_type,i)] = [y,h,x,w]
        cv2.putText(initial_image,"#{}".format(i+1),(x,y-15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
    try:
        os.mkdir(folder+"/Annotated")
    except:
        pass
    cv2.imwrite(folder+"/Annotated/%s_annotated.png"%cell_type,initial_image)
    return crop_rect, spot_loc

    
def find_pixels_in_CFUs(image_f,rects):
    #thresh = binarize_image(image_f)
    #mask = mask_image(thresh)
    mask = cv2.imread(image_f,0)
    pixels = OrderedDict()
    all_pix_size = []
    for n,i in rects.items():
        pix = mask[i[0]:i[0]+i[1],i[2]:i[2]+i[3]]
        all_pix_size.append(pix.size)
    for n,i in rects.items():
        pix = mask[i[0]:i[0]+i[1],i[2]:i[2]+i[3]]
        pixels[n] = np.sum(pix)#(np.count_nonzero(pix)/float(pix.size))*(float(pix.size)/max(all_pix_size))
    return pixels


def get_rects(image,cell_type,size, folder):
    thresh = binarize_image(image)
    mask = mask_image(thresh,size)
    return find_CFUs(mask, image, cell_type, folder)
    

def get_growth(folder,cell_types,max_times,size):
    growth = pd.DataFrame()
    for i, max_time in zip(cell_types,max_times):
        rects, spot_loc = get_rects("%s/%s_%s.png"%(folder,i,max_time),i,size, folder.split("/")[0])
        for files in os.listdir(folder):
            cell_type, time = files.split("_")
            if cell_type == i:
                pixels = find_pixels_in_CFUs("%s/%s"%(folder,files),rects)
                for k,z in pixels.items():
                    index = "%s_%s"%(k,time.split(".")[0])
                    growth.at[index,"Cell_type"] = cell_type
                    growth.at[index,"Time"] = int(time.split(".")[0])
                    growth.at[index,"Spot"] = k
                    growth.at[index,"Spot_loc"] = spot_loc[k]
                    growth.at[index,"Growth"] = z
    return growth
                    
def convert_dataframe_to_matrix(df, row, column, values):
    matrix = pd.DataFrame()
    for n, w in df.iterrows():
        matrix.at[w[row],w[column]] = w[values]
    return matrix

def plot_time_vs_growth(data,y, cell_types,colors):
    for n,(i,c) in enumerate(zip(cell_types,colors)):
        sub_data = data[data["Cell_type"].isin([i])]
        sns.regplot(x="Time",y = y,data = sub_data,x_estimator=np.mean,truncate = True, order = 3, fit_reg = True, color = c)
    plt.show()


def normalize_to_spots(df):
    spot_group = df.groupby("Spot")
    time_zero = {}
    for i, group in spot_group:
        time_zero[i] = group.loc[group["Time"]==0]["Growth"].values
    for n, rows in df.iterrows():
        norm_growth = (rows["Growth"]-time_zero[rows["Spot"]])
        if norm_growth > 0:
            df.at[n,"Growth_normalized"]= float(norm_growth)
        else:
            df.at[n,"Growth_normalized"]= 0.0
    return df

def norm_normalize_to_spots(df):
    spot_group = df.groupby("Spot")
    max_pix = {}
    for i, group in spot_group:
        max_pix[i] = max(group["Growth_normalized"].values)
    for n, rows in df.iterrows():
        if rows["Growth_normalized"] > 0:
            norm_norm_growth = (rows["Growth_normalized"]/float(max_pix[rows["Spot"]]))
            df.at[n,"Growth_normalized"]= float(norm_norm_growth)
        else:
            df.at[n,"Growth_normalized"]= 0.0
    return df

def gompz_growth_curve(x,A,gr,lp,e): # This is the Gompertz equation
    return A*np.exp(-np.exp((((gr*e)/A)*(lp-x))+1))

def fit_curve(function,x,y): # Fits the curve
    try:
        popt,pcov = optimize.curve_fit(function,x,y, maxfev = 1000)
    except:
        popt = np.array([0.,0.,0.,0.])
    return popt

def get_growth_properties(df):
    growth_properties = pd.DataFrame()
    grouped = df.groupby("Spot")
    for name, group in grouped:
        A,G,L,e = fit_curve(gompz_growth_curve,group["Time"],group["Normed_Growth_normalized"])
        growth_properties.at[name,"Cell_type"] = name.split("_")[0]
        growth_properties.at[name,"Max_growth"] = A
        growth_properties.at[name,"Growth_rate"] = G
        growth_properties.at[name,"Lag_phase"] = L
    return growth_properties

def get_time_of_appearance(df,pixden):
    TOA = pd.DataFrame()
    grouped = df.groupby("Spot").apply(lambda x: x.sort_values(['Time']))
    group = grouped.groupby("Spot")
    for nm,gr in group:
        for ii,kk in gr.iterrows():
            if kk["Growth_normalized"] >1000:
                TOA.at[nm,"TOA"] = kk["Time"]
                TOA.at[nm,"Cell_type"] = kk["Cell_type"]
                TOA.at[nm,"Spot_loc"] = kk["Spot_loc"]
                break
            else:
                pass
    return TOA

def plot_growth_prop_dist(data, cell_type,colors):
    fig, axs = plt.subplots(1,3, figsize=(15, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    for z,(j,c) in enumerate(zip(cell_type,colors)):
        sub_data = data[data["Cell_type"].isin([j])]
        for n,(i,axis) in enumerate(zip(["Max_growth","Growth_rate","Lag_phase"],axs)):
            dq90= sub_data[i].quantile(0.9)
            sub_sub_data= sub_data[sub_data[i].between(0,dq90, inclusive=False)][i]
            sns.distplot(sub_sub_data, label=i, color = c, ax = axis, norm_hist = True)
    plt.show()


def plot_toa_dist(data, cell_types,colors):
    for n,(i,c) in enumerate(zip(cell_types,colors)):
        sub_data = data[data["Cell_type"].isin([i])]["TOA"]
        bins = np.arange(0,max(data["TOA"].values),90)
        sns.distplot(sub_data,label = "%s(%s)"%(i,sub_data.size),color = c,norm_hist = False, kde = True, bins = bins)
    plt.legend()
    plt.xlabel("Time of appearance 12hrs + (Minutes)",fontsize = 18)
    plt.ylabel("Frequency of CFUs",fontsize = 18)
    #plt.grid(b = None,which = "major",axis="both")
    plt.savefig("%s.pdf"%raw_input("File name"))
    plt.show()
    
def mark_spots(TOA,image,parent_folder,cell_type,ranges):
    """Ranges should be order of lower range and upper range"""
    image_read = cv2.imread(image,0)
    initial_image = cv2.cvtColor(image_read,cv2.COLOR_GRAY2BGR)
    for i,c in TOA.iterrows():
        if c["Cell_type"] == cell_type:
            (cX,cY,radius) = (int(aa) for aa in c["Spot_loc"].split("_"))
            if c["TOA"] in range(ranges[0][0],ranges[0][1]):
                color = color_arrays["G"]
            elif c["TOA"] in range(ranges[1][0],ranges[1][1]):
                color = color_arrays["R"]
            else:
                color = color_arrays["B"]
            cv2.circle(initial_image,(int(cX),int(cY)),int(radius),
                       color,-1)
            cv2.putText(initial_image,"#{}".format(i+1),(cX,cY-15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
    try:
        os.mkdir(parent_folder+"/TOA_marked")
    except:
        pass
    cv2.imwrite(parent_folder+"/TOA_marked/%s_TOAmarked.png"%cell_type,initial_image)

circle = np.array([[1672,856,550],[3340,876,550],[1604,2584,550],[3328,2608,550],[1333,4313,550],[3224,4317,550]])
rectangle = np.array([[2588,3620],[3240,894],[2028,2100],[909,3356],[870,858]])
name = [i for i in "BCDE"]
time = [2010,2620,2620,2132]
color_arrays = {"R":(0,0,255),"G":(0,255,0),"B":(255,0,0)}
