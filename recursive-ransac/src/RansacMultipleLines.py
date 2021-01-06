import numpy as np
from matplotlib import pyplot as plt

from skimage.measure import LineModelND, ransac
import os
from skimage import io
import math
import datetime

min_samples=3 #RANSAC parameter - The minimum number of data points to fit a model to.
#min_inliers_allowed=5 #Custom parameter  - A line is selected only if these many inliers are found

def read_black_pixels(imagefilename:str):
    #returns a numpy array with shape (N,2) N points, x=[0], y=[1]
    #The coordinate system is Cartesian
    np_image=io.imread(imagefilename,as_gray=True)
    black_white_threshold=0
    if (np_image.dtype == 'float'):
        black_white_threshold=0.5
    elif (np_image.dtype == 'uint8'):
        black_white_threshold=128
    else:
        raise Exception("Invalid dtype %s " % (np_image.dtype))
    indices=np.where(np_image <= black_white_threshold)
    width=np_image.shape[1]
    height=np_image.shape[0]
    cartesian_y=height-indices[0]-1
    np_data_points=np.column_stack((indices[1],cartesian_y)) 
    return np_data_points, width,height

def extract_first_ransac_line(data_points:[], max_distance:int):
    #Accepts a numpy array with shape N,2  N points, with coordinates x=[0],y=[1]
    #Returns a numpy array with shape (N,2), these are the inliers of the just discovered ransac line
    
    
    model_robust, inliers = ransac(data_points, LineModelND, min_samples=min_samples,
                                   residual_threshold=max_distance, max_trials=1000)
    results_inliers=[]
    results_inliers_removed=[]
    for i in range(0,len(data_points)):
        if (inliers[i] == False):
            #Not an inlier
            results_inliers_removed.append(data_points[i])
            continue
        x=data_points[i][0]
        y=data_points[i][1]
        results_inliers.append((x,y))
    return np.array(results_inliers), np.array(results_inliers_removed)


def superimpose_all_inliers(points_lists,width:float, height:float):
    #Create an RGB image array with dimension heightXwidth
    #Draw the points with various colours
    #return the array

    new_image=np.full([height,width,3],255,dtype='int')
    colors=[(0,255,0),(255,255,0),(0,0,255)]
    for points_index in range(0,len(points_lists)):
        color=colors[points_index % len(colors)]
        points_array=points_lists[points_index]
        for point in points_array:
            x=point[0]
            y=point[1]
            new_y=height-y-1
            new_image[new_y][x][0]=color[0]
            new_image[new_y][x][1]=color[1]
            new_image[new_y][x][2]=color[2]
    return new_image

def extract_multiple_lines_and_save(inputfilename:str,iterations:int, max_distance:int,min_inliers_allowed:int):
    """
    min_inliers_allowed - a line is selected only if it has more than this inliers. The search process is halted when this condition is met
    max_distance - This is the RANSAC threshold distance from a line for a point to be classified as inlier
    """
    print("---------------------------------------")
    print("Processing: %s" % (inputfilename))
    folder_script=os.path.dirname(__file__)
    absolute_path=os.path.join(folder_script,"images/",inputfilename)

    results=[]
    all_black_points,width,height=read_black_pixels(absolute_path)
    print("Found %d pixels in the file %s" % (len(all_black_points),inputfilename))
    starting_points=all_black_points
    for index in range(0,iterations):
        if (len(starting_points) <= min_samples):
            print("No more points available. Terminating search for RANSAC")
            break
        inlier_points,inliers_removed_from_starting=extract_first_ransac_line(starting_points,max_distance=max_distance)
        if (len(inlier_points) < min_inliers_allowed):
            print("Not sufficeint inliers found %d , threshold=%d, therefore halting" % (len(inlier_points),min_inliers_allowed))
            break
        starting_points=inliers_removed_from_starting
        results.append(inlier_points)
        print("Found %d RANSAC lines" % (len(results)))
    superimposed_image=superimpose_all_inliers(results,width,height)
    #Save the results
    filename_noextension=os.path.splitext(inputfilename)[0]
    folder_script=os.path.dirname(__file__)
    file_result=os.path.join(folder_script,"./out/",("result-%s.png") % (filename_noextension))
    io.imsave(file_result,superimposed_image)
    print("Results saved to file %s" % (file_result))




# extract_multiple_lines_and_save("SmallCross.png",5)
# extract_multiple_lines_and_save("SmallCrossWithNoise.png",5)
# extract_multiple_lines_and_save("2ProminentLine.png",5)
#todo some problem in one of the above lines, excpetion
extract_multiple_lines_and_save("2ProminentLineWithNoise.png",5,max_distance=3, min_inliers_allowed=5)
extract_multiple_lines_and_save("3ProminentLineWithNoise.png",5, max_distance=3,min_inliers_allowed=5)
