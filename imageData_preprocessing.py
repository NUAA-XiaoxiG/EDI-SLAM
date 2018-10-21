#------ modified by xXiaoxi Gong 2018/05/21 ------#
import os
import cv2
import math
import numpy as np
import random
from matplotlib import pyplot as plt

### Make a new direct 'dataset' and to unzip all
### TUM datasets here.
def imgNum_count(file_path, eventImg_path):
    if (os.path.exists(eventImg_path) != True):
        os.makedirs(eventImg_path)
    img_num_count = []
    img_files_count = []

    img_files = os.listdir(file_path)
    img_files_count.append(img_files)
    ### File number counts
    for i in img_files:
        name = file_path + img_files[img_files.index(i)] + '/rgb/'
        if (os.path.exists(name)): # if the path exists...
            img_num_count.append(len(os.listdir(name)))
        else:
            print('No such files or path: ' + name)
    return img_num_count, img_files_count

### First 3 parameters represent image pairs' difference, frame at 't' time
### and frame at 't+1' time. The last two parameters control band pass filter's
### downside and upside threshold value, respectively.
def evt_img_generator(df_img, t0_img, t1_img, filt_ds, filt_us):
    ### Image size of TUM datasets
    img_Height = 480
    img_Width = 640
    img_Channel = 3
    evt_img = np.zeros((img_Height,img_Width,img_Channel), dtype="uint8")
    for j in range(img_Width):
        for i in range(img_Height):
            if ( t1_img[i, j] >= t0_img[i, j]
                 and abs(df_img[i, j]) >= filt_ds
                 and abs(df_img[i, j]) <= filt_us):
                evt_img[i, j, 0] = df_img[i, j]
                evt_img[i, j, 1] = 0
                evt_img[i, j, 2] = 0
            if ( t1_img[i, j] <= t0_img[i, j]
                and abs(df_img[i, j]) >= filt_ds
                and abs(df_img[i, j]) <= filt_us):
                evt_img[i, j, 0] = 0
                evt_img[i, j, 1] = 0
                evt_img[i, j, 2] = df_img[i, j]
    return evt_img
#evt_img_generator()

### 1305032350.7780 0.9650 -0.3636 1.2560 0.3495 0.6802 -0.6064 -0.2177
def quadra_2_eulclide(input):
    ts, tx, ty, tz, qx, qy, qz, qw = input.split(' ', 8)
    ts = float(ts)
    tx = float(tx)
    ty = float(ty)
    tz = float(tz)
    qx = float(qx)
    qy = float(qy)
    qz = float(qz)
    qw = float(qw)

    # x_r = arctan( 2(wx+yz)/(1-2(x^2+y^2)) )
    # y_r = arcsin( 2(wy-zx) )
    # z_r = arctan( 2(wz+xy)/(1-2(y^2+z^2)) )
    ax = math.atan( 2*(qw*qx + qy*qz)/(1 - 2*(qx*qx + qy+qy)) )
    ay = math.asin( 2*(qw*qy - qz*qx) )
    az = math.atan( 2*(qw*qz + qx*qy)/(1 - 2*(qy*qy + qz*qz)) )

    output = ( str(ts) + ' ' + str(tx) + ' ' + str(ty) + ' ' + str(tz) + ' ' + str( round(ax,4) ) + ' ' + str( round(ay,4) ) + ' ' + str( round(az,4) ) )
    return output

def img_regularization(img_input, resize_ratio):
    img_h = len(img_input[:,0])
    img_w = len(img_input[0,:])
    img_resize = cv2.resize(img_input, ( int(img_w/resize_ratio), int(img_h/resize_ratio) ), interpolation=cv2.INTER_AREA)
    img_average = round( np.average(img_resize), 2 )
    return img_average

def filter_area(diffImg, EDI):
    sum_DI = np.sum(diffImg)
    sum_EDI = np.sum(EDI)
    ratio = round( (sum_EDI/sum_DI)*100, 2)
    return ratio

### Generate event-based difference image and related ground truth
def images_gndTruth_registration(path_for_origin_datasets, path_for_processed_data, frame_interval):
    flag_01 = 0
    flag_02 = 0
    itr_end = 36
    imgNum_cnt, imgFiles_cnt = imgNum_count(path_for_origin_datasets, path_for_processed_data + 'rgb/')
    gnd_diff = open(path_for_processed_data + 'gnd_diff.txt', 'a')
    for i in range( len(imgNum_cnt) ):
        itr_start = 1
        itr_cnt = 0
        imgFiles_cnt_list = imgFiles_cnt[0]
        imgFiles_cnt_str = imgFiles_cnt_list[i]

        file_rgb = open(path_for_origin_datasets + imgFiles_cnt_str + '/rgb.txt', 'r')
        file_gnd = open(path_for_origin_datasets + imgFiles_cnt_str + '/groundtruth.txt', 'r')
        line_gnd = file_gnd.readlines()
        line_rgb = file_rgb.readlines()

        ### To register images and ground truth here
        ### gnd_diff = open(path_for_processed_data + 'gnd_diff.txt', 'a')
        for j2 in range(3, len(line_rgb) - frame_interval):
            for j1 in range(3, len(line_gnd)):  # gnd_mat[:,0]
                line_rgb_str_01 = line_rgb[j2]
                img_timeStamp_01, img_pngFile_01 = line_rgb_str_01.split(' ', 1)
                line_gnd_str_01 = line_gnd[j1]
                gnd_timeStamp_01, gnd_pngFile_01 = line_gnd_str_01.split(' ', 1)
                imgPath_01 = path_for_origin_datasets + imgFiles_cnt_str + '/rgb/' + img_timeStamp_01 + '.png'

                line_rgb_str_02 = line_rgb[j2 + frame_interval]
                img_timeStamp_02, img_pngFile_02 = line_rgb_str_02.split(' ', 1)
                line_gnd_str_02 = line_gnd[j1]
                gnd_timeStamp_02, gnd_pngFile_02 = line_gnd_str_02.split(' ', 1)
                imgPath_02 = path_for_origin_datasets + imgFiles_cnt_str + '/rgb/' + img_timeStamp_02 + '.png'

                if (abs( float(img_timeStamp_01) - float(gnd_timeStamp_01) ) < 0.01 and os.path.exists(imgPath_01) == True):
                    gnd_reg_01 = line_gnd_str_01
                    gnd_trans_01 = quadra_2_eulclide(line_gnd_str_01)
                    flag_01 = 1

                if (abs( float(img_timeStamp_02) - float(gnd_timeStamp_02) ) < 0.01 and os.path.exists(imgPath_02) == True):
                    gnd_reg_02 = line_gnd_str_02
                    gnd_trans_02 = quadra_2_eulclide(line_gnd_str_02)
                    flag_02 = 1

                ### EDI calculation
                if(flag_01 == 1 and flag_02 == 1):
                    ### Difference of ground truth calculation
                    ### rgb_and_gnd_register.write(img_timeStamp + ' ' + gnd_timeStamp + ' ' + gnd_pngFile) #'\n'
                    ts01, tx01, ty01, tz01, ax01, ay01, az01 = gnd_trans_01.split(' ', 7)
                    ts02, tx02, ty02, tz02, ax02, ay02, az02 = gnd_trans_02.split(' ', 7)
                    tsDf = float(ts01) - float(ts02)
                    txDf = float(tx01) - float(tx02)
                    tyDf = float(ty01) - float(ty02)
                    tzDf = float(tz01) - float(tz02)
                    axDf = float(ax01) - float(ax02)
                    ayDf = float(ay01) - float(ay02)
                    azDf = float(az01) - float(az02)
                    gnd_diff.write( img_timeStamp_02 + ' ' + str( round(tsDf,2) ) + ' ' +
                                    str( round(txDf,4) ) + ' ' + str( round(tyDf,4) ) + ' ' + str( round(tzDf,4) ) + ' ' +
                                    str( round(axDf,4) ) + ' ' + str( round(ayDf,4) ) + ' ' + str( round(azDf,4) ) + '\n' )

                    ### Difference image calculation
                    imgPath_01 = path_for_origin_datasets + imgFiles_cnt_str + '/rgb/' + img_timeStamp_01 + '.png'
                    imgPath_02 = path_for_origin_datasets + imgFiles_cnt_str + '/rgb/' + img_timeStamp_02 + '.png'

                    img_01 = cv2.imread(imgPath_01, cv2.IMREAD_GRAYSCALE)
                    img_02 = cv2.imread(imgPath_02, cv2.IMREAD_GRAYSCALE)
                    ### Intensity balance
                    lightInt = int(img_regularization(img_02, 10) - img_regularization(img_01, 10))
                    img_02 = img_02 - lightInt
					
					### Image normalization
					img_01 = img_01 / 127.5 - 1.0
                    img_02 = img_02 / 127.5 - 1.0
                    img_diff = abs( img_02[:, :] - img_01[:, :] ) * 127.5

                    #img_event = evt_img_generator(img_diff, img_01, img_02, 5, 235)
                    imgPath_evt = path_for_processed_data + 'rgb/' + img_timeStamp_02 + '.png'

                    ### filter noise-like pixels as the ratio
                    for shift_ratio in range(itr_start, itr_end):
                        img_event = evt_img_generator(img_diff, img_01, img_02, int(0 + shift_ratio*0.6), 255 - shift_ratio)
                        Filter = filter_area(img_diff, img_event)
                        print('Iterate times: ' + str(shift_ratio) + ' / 35 ... Filter ratio: ' + str(Filter) + ' ... Intensity var: ' + str(lightInt) )
                        if (Filter <= 5 or shift_ratio >= itr_end - 1):
                            if (Filter > 2):
                                itr_start = shift_ratio - 1
                            itr_cnt = itr_cnt + 1
                            if(itr_cnt == 10):
                                itr_start = 1
                                itr_cnt = 0
                            cv2.imwrite(imgPath_evt, img_event)
                            #print(str(Filter)+' '+str(shift_ratio)+' '+str(itr_cnt))
                            break
                    #------------------#
                    flag_01 = 0
                    flag_02 = 0
                    break
            print(imgFiles_cnt_str + ': ' + str(len(line_rgb) - 4) + ' / ' + str(j2 - 2) )
    gnd_diff.close()
images_gndTruth_registration('D:/datasets/', 'D:/datasets_pre/', 1)
