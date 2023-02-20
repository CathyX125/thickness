import cv2
import numpy as np
import sys
import os
import math
import re
from matplotlib import pyplot as plt


def oxide_thickness(file_name, show_debug_steps=False, show_debug_contours=False, debug_thickness=False):
    slice_size = 40
    sample_density = 3
    remove_noise = False
    max_noise_size = 30
    one_over_percent = 50 # 2%
    bottom_discard = 0
    black_level = 80
    grey_level = 180
    max_padding_top = 80
    padding = 15
    legend_height = 160

    #### Read image ####
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    if show_debug_contours or show_debug_steps or debug_thickness:
        print ("image size: {} x {}".format(height, width))

    #### Thresholding ####
    img = img[0:height-legend_height-bottom_discard, 0:width]
    height, width = img.shape

    if show_debug_steps:
        cv2.imshow('image',img)
        cv2.waitKey(0)

    # con = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #            cv2.THRESH_BINARY, 55, 2)

    _, con = cv2.threshold(img, black_level, 255, cv2.THRESH_BINARY)

    if show_debug_steps:
        cv2.imshow("thresholded", con)
        cv2.waitKey(0)

    #### MSER to find contours ####

    con = cv2.bitwise_not(con)
    mser = cv2.MSER_create()
    mask = np.zeros((con.shape[0], con.shape[1], 1), dtype=np.uint8)

    bonding_x = width
    bonding_y = height
    bonding_x2 = 0
    bonding_y2 = 0

    cnts = []
    for x in range(slice_size):
        x_start = math.floor(width*x/slice_size-width/slice_size)
        x_end = math.ceil(width*(x+1)/slice_size+width/slice_size+2)
        if x_start < 0:
            x_start = 0
        if x_end >= width:
            x_end = width-1

        # fine mser regions
        regions, _ = mser.detectRegions(con[0:height, x_start:x_end])
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        for contour in hulls:
            contour+=(x_start, 0)

        # ignore smaller regions
        for contour in hulls:
            if cv2.contourArea(contour) < width*height/one_over_percent/slice_size:
                continue

            # find the bonding box of the oxid
            x,y,w,h = cv2.boundingRect(contour)
            if x<bonding_x: bonding_x=x
            if y<bonding_y: bonding_y=y
            if x+w>bonding_x2: bonding_x2=x+w
            if y+h>bonding_y2: bonding_y2=y+h

            cv2.drawContours(mask, [contour], -1, 255, -1)

    mser_d = cv2.bitwise_and(con, con, mask=mask)
    mat = cv2.bitwise_not(mser_d)

    bonding_x -= padding
    bonding_x2 += padding
    bonding_y -= padding
    bonding_y2 += padding
    if bonding_x<0:bonding_x=0
    if bonding_x2>=width:bonding_x2=width-1
    if bonding_y<0:bonding_y=0
    if bonding_y2>=height:bonding_y2=height-1

    mat = mat[bonding_y:bonding_y2, bonding_x:bonding_x2]
    height, width = mat.shape

    bonding_y_t = bonding_y - max_padding_top
    if bonding_y - max_padding_top < 0:
        bonding_y_t = 0

    img_pad = img[bonding_y_t:bonding_y2, bonding_x:bonding_x2]

    if show_debug_steps:
        cv2.imshow("mser", mat)
        cv2.waitKey(0)


    #### Remove any area < max_noise_size in the bonding box ####
    if remove_noise:
        # mat = [
        # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # [1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        # [1, 1, 0, 1, 0 ,1, 1, 1, 0, 1],
        # [1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        # [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        # [1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
        # [1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
        # [1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        # [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        # ]
        # height, width = 10, 10
        sys.setrecursionlimit(1000)
        q = []
        def aju(x, y):
            if (x, y-1) in q or y-1 < 0 or mat[y-1][x]:
                return 0
            return 1

        def ajd(x, y):
            if (x, y+1) in q or y+1 >= height or mat[y+1][x]:
                return 0
            return 1

        def ajl(x, y):
            if (x-1, y) in q or x-1 < 0 or mat[y][x-1]:
                return 0
            return 1

        def ajr(x, y):
            if (x+1, y) in q or x+1 >= width or mat[y][x+1]:
                return 0
            return 1

        def findaj(x, y, from_d):
            q.append((x, y))
            if len(q) > max_noise_size:
                return
            if from_d != 'r' and ajr(x, y):
                findaj(x + 1, y, 'l')
            if from_d != 'd' and ajd(x, y):
                findaj(x, y + 1, 'u')
            if from_d != 'l' and ajl(x, y):
                findaj(x - 1, y, 'r')
            if from_d != 'u' and aju(x, y):
                findaj(x, y - 1, 'd')

        for ix in range(width):
            for iy in range(height):
                if mat[iy][ix] == 0:
                    q = []
                    findaj(ix, iy, 'o')
                    if len(q) and len(q) < max_noise_size:
                        for (x, y) in q:
                            mat[y][x] = 255
    # end if remove_noise

    #### Calculate thickness and length of the oxid in pixels in the image ###

    upbond = []
    lowbond = []

    slice_size *= sample_density
    cnts = []
    slice_lengths = []
    slice_lengths_center = []
    slice_lengths_top = []

    for i in range(slice_size):
        x_start = math.floor(width*i/slice_size-+width/slice_size/2)
        x_end = math.ceil(width*(i+1)/slice_size+width/slice_size/2+2)
        if x_start < 0:
            x_start = 0
        if x_end >= width:
            x_end = width-1
        slice_width = x_end - x_start
        center_line_x = []
        center_line_y = []
        top_line_x = []
        top_line_y = []
        for ix in range(x_start, x_end):
            ty = 0
            by = 0
            found = False
            for iy in range(height):
                if mat[iy][ix]==0:
                    upbond.append([[ix, iy]])
                    ty = iy
                    found = True
                    break
            for iy in reversed(range(height)):
                if mat[iy][ix]==0:
                    lowbond.append([[ix, iy]])
                    by = iy
                    found = True
                    break
            if found:
                center_line_x.append(ix)
                center_line_y.append((ty+by)/2)
                top_line_x.append(ix)
                top_line_y.append(ty)

        if len(center_line_x) > (slice_width/2):
            # LSTSQ for line fitting
            A = np.vstack([center_line_x, np.ones(len(center_line_x))]).T
            # y = mx + c
            m, c = np.linalg.lstsq(A, center_line_y, rcond=None)[0]

            # length of line in the slice
            slice_length = slice_width * math.sqrt(m**2 + 1)
            slice_lengths.append(slice_length)

            slice_length_center = math.sqrt((center_line_x[0] - center_line_x[-1])**2 + (center_line_y[0] - center_line_y[-1])**2 )
            slice_lengths_center.append(slice_length_center)

            # if show_debug_contours or show_debug_steps:
            #     cv2.line(img, (x_start, int(m*x_start+c)+bonding_y), (x_end, int(m*x_end+c)+bonding_y), 125, 1)
            #     cv2.line(img, (x_start, int(center_line_y[0])+bonding_y), (x_end, int(center_line_y[-1])+bonding_y), 255, 1)

            cnt = np.array(upbond+lowbond[::-1], dtype=np.int32)
            cnts.append(cnt)

        if len(top_line_x) > (slice_width/2):
            # LSTSQ for line fitting
            A = np.vstack([top_line_x, np.ones(len(top_line_x))]).T
            # y = mx + c
            m, c = np.linalg.lstsq(A, top_line_y, rcond=None)[0]

            # length of line in the slice
            slice_length = slice_width * math.sqrt(m**2 + 1)
            # if show_debug_steps:
            #     print(slice_width, slice_length)
            slice_lengths_top.append(slice_length)

            if show_debug_contours or show_debug_steps:
                cv2.line(img, (x_start, int(m*x_start+c)+bonding_y), (x_end, int(m*x_end+c)+bonding_y), 255, 1)

    if show_debug_contours or show_debug_steps:
        contours = np.zeros((height, width, 1), dtype=np.uint8)
        for cnt in cnts:
            cv2.drawContours(contours,[cnt],0,255,1)
        cv2.imshow('contours',contours)
        cv2.waitKey(0)

    line_length_sum = 0
    for i in range(len(slice_lengths)):
        if i == 0 or i == len(slice_lengths) - 1:
            line_length_sum += slice_lengths[i]/ 3 * 2
        else:
            line_length_sum += slice_lengths[i]/ 2

    center_line_length_sum = 0
    for i in range(len(slice_lengths_center)):
        if i == 0 or i == len(slice_lengths_center) - 1:
            center_line_length_sum += slice_lengths_center[i]/ 3 * 2
        else:
            center_line_length_sum += slice_lengths_center[i]/ 2
    
    top_line_length_sum = 0
    for i in range(len(slice_lengths_top)):
        if i == 0 or i == len(slice_lengths_top) - 1:
            top_line_length_sum += slice_lengths_top[i]/ 3 * 2
        else:
            top_line_length_sum += slice_lengths_top[i]/ 2

    if show_debug_contours or show_debug_steps or debug_thickness:
        print("Oxid total regressed line length in pixels = {}".format(line_length_sum))
        print("Oxid total center line length in pixels = {}".format(center_line_length_sum))
        print("Oxid total top line length in pixels = {}".format(top_line_length_sum))

    if show_debug_contours or show_debug_steps or debug_thickness:
        cv2.imshow('result', img_pad)
        cv2.waitKey(0)

    hist, bins = np.histogram(img_pad.ravel(), int(grey_level+black_level/2),[0, int(grey_level+black_level/2)])
    max_hist = 0
    for i in range(0, black_level):
        if hist[i] > max_hist:
            max_hist = hist[i]
            max_bin = bins[i]

    black_center = max_bin
    max_hist = 0

    for i in range(black_level, grey_level):
        if hist[i] > max_hist:
            max_hist = hist[i]
            max_bin = bins[i]
    grey_center = max_bin

    black_count = 0
    grey_count = 0
    for i in range(max(0, int(black_center-black_level/2)), int(black_center+black_level/2)):
        black_count += hist[i]

    for i in range(max(0, int(grey_center-black_level/2)), int(grey_center+black_level/2)):
        grey_count += hist[i]

    if debug_thickness:
        plt.hist(img_pad.ravel(), int(grey_level+black_level/2),[0, int(grey_level+black_level/2)])
        plt.show()
        print(black_count, grey_count)

    return top_line_length_sum, black_count, grey_count


# Simplified for faster process
def fast_oxide_thickness(file_name, show_debug_steps=False, show_debug_contours=False, debug_thickness=False):
    slice_size = 12
    sample_density = 10
    bottom_discard = 0
    black_level = 80
    grey_level = 180
    max_padding_top = 80
    padding = 15
    legend_height = 160

    #### Read image ####
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    if show_debug_contours or show_debug_steps or debug_thickness:
        print ("image size: {} x {}".format(height, width))

    img = img[0:height-legend_height-bottom_discard, 0:width]
    height, width = img.shape

    if show_debug_steps:
        cv2.imshow('image',img)
        cv2.waitKey(0)

    #### Thresholding ####

    # con = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #            cv2.THRESH_BINARY, 55, 2)

    _, con = cv2.threshold(img, black_level, 255, cv2.THRESH_BINARY)

    if show_debug_steps:
        cv2.imshow("thresholded", con)
        cv2.waitKey(0)

    #### MSER to find contours ####

    con = cv2.bitwise_not(con)
    mser = cv2.MSER_create()
    mask = np.zeros((con.shape[0], con.shape[1], 1), dtype=np.uint8)

    bonding_x = width
    bonding_y = height
    bonding_x2 = 0
    bonding_y2 = 0

    for x in range(slice_size):
        x_start = math.floor(width*x/slice_size-width/slice_size)
        x_end = math.ceil(width*(x+1)/slice_size+width/slice_size+2)
        if x_start < 0:
            x_start = 0
        if x_end >= width:
            x_end = width-1

        # fine mser regions
        regions, _ = mser.detectRegions(con[0:height, x_start:x_end])
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        for contour in hulls:
            contour+=(x_start, 0)

        # ignore smaller regions
        for contour in hulls:
            # find the bonding box of the oxid
            x,y,w,h = cv2.boundingRect(contour)
            if x<bonding_x: bonding_x=x
            if y<bonding_y: bonding_y=y
            if x+w>bonding_x2: bonding_x2=x+w
            if y+h>bonding_y2: bonding_y2=y+h

            cv2.drawContours(mask, [contour], -1, 255, -1)

    mser_d = cv2.bitwise_and(con, con, mask=mask)
    mat = cv2.bitwise_not(mser_d)

    bonding_x -= padding
    bonding_x2 += padding
    bonding_y -= padding
    bonding_y2 += padding
    if bonding_x<0:bonding_x=0
    if bonding_x2>=width:bonding_x2=width-1
    if bonding_y<0:bonding_y=0
    if bonding_y2>=height:bonding_y2=height-1

    mat = mat[bonding_y:bonding_y2, bonding_x:bonding_x2]
    height, width = mat.shape

    bonding_y_t = bonding_y - max_padding_top
    if bonding_y - max_padding_top < 0:
        bonding_y_t = 0

    img_pad = img[bonding_y_t:bonding_y2, bonding_x:bonding_x2]

    if show_debug_steps:
        cv2.imshow("mser", mat)
        cv2.waitKey(0)

    #### Calculate thickness and length of the oxid in pixels in the image ###

    upbond = []
    lowbond = []

    slice_size *= sample_density
    cnts = []
    slice_lengths_top = []

    for i in range(slice_size):
        x_start = math.floor(width*i/slice_size-+width/slice_size/2)
        x_end = math.ceil(width*(i+1)/slice_size+width/slice_size/2+2)
        if x_start < 0:
            x_start = 0
        if x_end >= width:
            x_end = width-1
        slice_width = x_end - x_start
        top_line_x = []
        top_line_y = []
        for ix in range(x_start, x_end):
            ty = 0
            by = 0
            found = False
            for iy in range(height):
                if mat[iy][ix]==0:
                    upbond.append([[ix, iy]])
                    ty = iy
                    found = True
                    break

            if found:
                top_line_x.append(ix)
                top_line_y.append(ty)

        if len(top_line_x) > (slice_width/2):
            # LSTSQ for line fitting
            A = np.vstack([top_line_x, np.ones(len(top_line_x))]).T
            # y = mx + c
            m, c = np.linalg.lstsq(A, top_line_y, rcond=None)[0]

            # length of line in the slice
            slice_length = slice_width * math.sqrt(m**2 + 1)
            # if show_debug_steps:
            #     print(slice_width, slice_length)
            slice_lengths_top.append(slice_length)

            if show_debug_contours or show_debug_steps:
                cv2.line(img, (x_start, int(m*x_start+c)+bonding_y), (x_end, int(m*x_end+c)+bonding_y), 255, 1)


    top_line_length_sum = 0
    for i in range(len(slice_lengths_top)):
        if i == 0 or i == len(slice_lengths_top) - 1:
            top_line_length_sum += slice_lengths_top[i]/ 3 * 2
        else:
            top_line_length_sum += slice_lengths_top[i]/ 2

    if show_debug_contours or show_debug_steps or debug_thickness:
        print("Oxid total top line length in pixels = {}".format(top_line_length_sum))

    if show_debug_contours or show_debug_steps or debug_thickness:
        cv2.imshow('result', img_pad)
        cv2.waitKey(0)

    hist, bins = np.histogram(img_pad.ravel(), int(grey_level+black_level/2),[0, int(grey_level+black_level/2)])
    max_hist = 0
    for i in range(0, black_level):
        if hist[i] > max_hist:
            max_hist = hist[i]
            max_bin = bins[i]

    black_center = max_bin
    max_hist = 0

    for i in range(black_level, grey_level):
        if hist[i] > max_hist:
            max_hist = hist[i]
            max_bin = bins[i]
    grey_center = max_bin

    black_count = 0
    grey_count = 0
    for i in range(max(0, int(black_center-black_level/2)), int(black_center+black_level/2)):
        black_count += hist[i]

    for i in range(max(0, int(grey_center-black_level/2)), int(grey_center+black_level/2)):
        grey_count += hist[i]

    if debug_thickness:
        plt.hist(img_pad.ravel(), int(grey_level+black_level/2),[0, int(grey_level+black_level/2)])
        plt.show()
        print(black_count, grey_count)

    return top_line_length_sum, black_count, grey_count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python {} file_name".format(sys.argv[0]))
        exit(128)

    file_name = sys.argv[1]
    oxide_thickness(file_name, True, True, True)
