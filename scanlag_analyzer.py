# -*- coding: utf-8 -*-
"""@Author: Vivek Srinivas - Baliga Lab, ISB
- This is a program to identify hetergeneity in growth.
- Requires Images folder with with time lapse images aquired on a flatbed documents scanner.
##refer to examples files for format##
    Step 1: Crop individual experiments
            >> crop_images_in_folder(folder_with_images,out_folder_for_cropped_images,plate_location_list,name_list,time0)
            ## folder for cropped images should be created before hand
            ## plate_location_list is a list of format :
                    ### plate_location_list = np.array([[x,y,550],....,[x,y,550]])
                    ### where x,y is the location of centers of plates obtained from imageJ
            ## name_list is the list if names to be given for the plates; should be in same order as the plate_location_list
            ## time0 is the time on the first image.
    Step 2: Calculate growth
            >> growth = get_growth(folder,cell_types,annotation_time_list,size)
            ## size can be 500 to 1500, should be decided based on the annotation
            ## annotation_time_list is the list of times for the experiments which needs to be annotated; this is obtained by checking through crop image folder to find the time for a given experiment where the colonies are distinct and visible.
    Step 3: Normalize to the spot
            >> growth_normalized_to_spots = normalize_to_spots(growth)
    Step 4: Save table
            ## save growth data to csv
            >> growth_normalized_to_spots = normalize_to_spots.to_csv(file_name)
    Step 5: Plot spot intensity vs time
            >> plot_time_vs_growth(growth_normalized_to_spots,"Growth_normalized",name_list,colors)
            ## identify spot intensity to be used as threshold
    Step 6: >> toa = get_time_of_appearance(growth_normalized_to_spots,pixel_threshold)
            ## threshold is ones identified in earlier step
    Step 7: Save TOA file
            >> toa.to_csv(file_name)
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
from scipy.stats import norm
from scipy import stats

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
    return circles

def crop_image(image_file,folder,circles,names,time):
    img = cv2.imread(image_file,0)
    img = cv2.medianBlur(img,5)
    cropped = []
    for c in circles:
        #y,x = c[0],c[1]
        #cropped.append(img[y:y+2000,x:x+2000])
        x,y,d = c[0],c[1],c[2]
        cropped.append(img[y-d:y+d,x-d:x+d])
    for n,cro in enumerate(cropped):
        cv2.imwrite("%s/%s_%s.png"%(folder,names[n],time),cro)

def crop_images_in_folder(folder,out_folder, circles,names,to):
    for i in os.listdir(folder):
        if i.split(".")[-1] == "png":
            t = int(i.split("_")[1].split(".")[0])
            crop_image("%s/%s"%(folder,i),out_folder,circles,names,int((t-to)/60.0))
        else:
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
        if numPixels > size[0] and numPixels < size[1]:
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
    """
    for n,i in rects.items():
        pix = mask[i[0]:i[0]+i[1],i[2]:i[2]+i[3]]
        all_pix_size.append(pix.size)"""
    for n,i in rects.items():
        pix = mask[i[0]:i[0]+i[1],i[2]:i[2]+i[3]]
        rect_area = abs((float(i[1])-i[0])*(float(i[3])-i[2]))
        total_pix = np.sum(pix)
        if rect_area > 0 and total_pix > 0:
            pixels[n] = total_pix#(np.count_nonzero(pix)/float(pix.size))*(float(pix.size)/max(all_pix_size))
        else:
            pixels[n] = 0.0
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
    ax = plt.subplot()
    groups = data.groupby("Cell_type")
    for gr,group in groups:
        if gr in cell_types:
            color = colors[cell_types.index(gr)]
            sns.regplot(x="Time",y = y,data = group,x_estimator=np.mean,truncate = False,fit_reg = False,\
                        color = color, label = gr,ax=ax)#, order = 3)
    plt.legend()
    plt.show()

def plot_growthrate_dist(data,y, cell_types,palette):
    sub_data2 = data[data["Cell_type"].isin(cell_types)]
    """ax2 = sns.violinplot(x="Time",y=y,data = sub_data2,\
                         scale_hue = True,inner = "stick",\
                         hue="Cell_type",scale = "count",\
                         split=True, palette = palette, cut = 0)
    for n,(i,c) in enumerate(zip(cell_types,colors)):
        sub_data = data[data["Cell_type"].isin([i])]
        ax = sns.violinplot(x = "Time",y = y, data = sub_data, scale = "count",color = c)
    """
    grid = sns.FacetGrid(sub_data2, row="Cell_type",col="Time", margin_titles=True, hue = "Cell_type",palette= palette)
    grid.map(plt.hist,y,bins = np.linspace(0,max(sub_data2[y].values),10))
    plt.legend()
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


def log_normalize_to_spots(df):
    spot_group = df.groupby("Spot")
    max_pix = max(df["Growth_normalized"].values)
    for n, rows in df.iterrows():
        if rows["Growth_normalized"] > 1:
            norm_norm_growth = np.log2(rows["Growth_normalized"])#(rows["Growth_normalized"]/float(max_pix))
            df.at[n,"Log_normalized"]= float(norm_norm_growth)
        else:
            df.at[n,"Log_normalized"]= 1.0
    return df

def gompertz_growth(time,x_scaling,y_scaling,max_growth):
    return (max_growth*(np.exp(-x_scaling*np.exp(-y_scaling*time))))

def get_lag(function,x,y): # Fits the curve
    try:
        popt,pcov = optimize.curve_fit(function,x,y, maxfev = 1000)
    except:
        popt = np.array([max(y),0.,0.])
    return popt[0]

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

def get_max_growth_rate(df):
    growth_properties = pd.DataFrame()
    grouped = df.sort_values(['Time'],ascending=True).groupby("Spot")
    for name, group in grouped:
        growth_rates = np.gradient(group["Growth_normalized"].values,2)
        growth_properties.at[name,"Cell_type"] = name.split("_")[0]
        growth_properties.at[name,"Max_growth"] = max(growth_rates)
        growth_properties.at[name,"Spot"] = name
    return growth_properties.set_index("Spot")

def get_growth_rate_at_tps(df):
    growth_properties = pd.DataFrame()
    grouped = df.sort_values(['Time'],ascending=True).groupby("Spot")
    spots = {}
    times = list(set(df.Time.values))
    times.sort()
    times_to_plot_index = range(0,len(times),len(times)/5)[2:]
    for name, group in grouped:
        growth_rates = np.gradient(group["Growth_normalized"].values,5)
        grt = [growth_rates[tts] for tts in times_to_plot_index]
        for grtt,ttss in zip(grt,times_to_plot_index):
                if grtt > 0:
                    index = "%s_%s"%(name,ttss)
                    growth_properties.at[index,"Cell_type"] = name.split("_")[0]
                    growth_properties.at[index,"Spot"] = name
                    growth_properties.at[index,"Growth_rate"] = grtt
                    growth_properties.at[index,"Time"] = times[ttss]
                else:
                    pass
    return growth_properties

def get_time_of_appearance(df,pixden):
    TOA = pd.DataFrame()
    group = df.groupby(["Spot","Cell_type","Spot_loc"])
    for (spot,ct,sl),gr in group:
        if gr.Growth_normalized.max() > pixden:
            gr = gr.sort_values(["Time"],ascending = True)
            for i, kk in gr.iterrows():
                if kk["Growth_normalized"] >=pixden:
                    TOA.at[spot,"TOA"] = kk["Time"]
                    break
                else :
                    pass
        else:
            TOA.at[spot,"TOA"] = gr["Time"].max()
        TOA.at[spot,"Cell_type"] = ct
        TOA.at[spot,"Spot_loc"] = sl
        TOA.at[spot,"Spot"] = spot
    return TOA.set_index("Spot")

def get_max_growth_and_toa(df,pixden):
    maxgrowth = get_max_growth_rate(df)
    toa = get_time_of_appearance(df,pixden)
    return pd.concat([toa,maxgrowth], axis=1, sort = False).reindex(toa.index)

def plot_toa_max_growthrate(data,cell_types,colors):
    fig = plt.figure()
    ax1 = plt.subplot2grid((4,4),(1,0),colspan=3,rowspan=3)
    ax2 = plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((4,4),(1,3),colspan=1,rowspan=3, sharey=ax1)
    for n,(i,c) in enumerate(zip(cell_types,colors)):
        sub_data = data[data["Cell_type"].isin([i])]
        toa_bins = np.arange(0,max(data["TOA"].values),30)
        max_bins = np.arange(0,max(data["Max_growth"].values),(max(data["Max_growth"].values)-min(data["Max_growth"].values))/50.0)
        sns.distplot(sub_data["TOA"],color = c,norm_hist = False, kde = True, bins = toa_bins, ax=ax2)
        sns.distplot(sub_data["Max_growth"],color = c,norm_hist = False, kde = True, bins = max_bins, ax=ax3, vertical=True)
        sns.scatterplot(x="TOA",y="Max_growth",data = sub_data, color = c, ax=ax1,label = "%s(%s)"%(i,sub_data.size))
    ax1.set_xlabel("Time of appearance 24hours + (Minutes)",fontsize = 14)
    ax1.set_ylabel("Maximum growth rate (Pixden/hour)",fontsize = 14)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    #plt.ylabel("Frequency of CFUs",fontsize = 18)
    #plt.grid(b = None,which = "major",axis="both")
    #plt.savefig("%s.pdf"%raw_input("File name"))
    plt.show()

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
        bins = np.arange(0,max(data["TOA"].values),60)
        sns.distplot(sub_data,label = "%s(%s)"%(i,sub_data.size),color = c,norm_hist = True, kde = True,bins = bins)
                     #hist_kws={"histtype": "step", "linewidth": 0.2,"alpha": 0.5, "color": c}, fit=norm)
    plt.legend()
    plt.xlabel("Time of appearance 12hrs + (Minutes)",fontsize = 18)
    plt.ylabel("Frequency of CFUs",fontsize = 18)
    #plt.grid(b = None,which = "major",axis="both")
    #plt.savefig("%s.pdf"%input("File name"))
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

def get_ks_stats(df,parameter):
    ret_dict = pd.DataFrame(columns=["Stats","pvalue"])
    group_dict = {}
    for gr, group in df.groupby("Cell_type"):
        group_dict[gr]=group[parameter].values
    for group1 in group_dict.keys():
        for group2 in group_dict.keys():
            ks = stats.ks_2samp(group_dict[group1],group_dict[group2])
            ret_dict.at["%s-%s"%(group1,group2),"Stats"]=ks[0]
            ret_dict.at["%s-%s"%(group1,group2),"pvalue"]=ks[1]
    return ret_dict

def plot_violin_plot(df,parameter, order, color):
    palette = {o:c for o,c in zip(order,color)}
    sns.set_style("whitegrid")
    ax = sns.boxplot(x="Cell_type",y=parameter,data = df,order=order,palette=color)#, inner = "quartile")
    #ax.set_ylim(df.TOA.min(),df.TOA.max())
    plt.show()

circle = np.array([[1398,2016,550],[2886,1116,550],[1410,3780,550],\
                   [2916,2892,550],[2922,4692,550]])
name = ["ArcAKO","TetR-ArcAKO-Pbr8","WT","TetR-ArcAKO","TetR"]
time = [1240,1378,1378,1700,2527]




