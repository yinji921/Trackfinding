
# coding: utf-8

# In[17]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
get_ipython().magic('matplotlib inline')


# In[18]:

def process_image(image):
        # NOTE: The output you return should be a color image (3 channel) for processing video below


    #reading in an image and turn it into grayscale
    #image = mpimg.imread('solidWhiteRight.jpg')
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #printing out some stats and plotting
    #print('This image is:', type(image), 'with dimensions:', image.shape)


    #Gaussian Blur and Canny!
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    low_threshold = 30
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask1 = np.zeros_like(edges)
    mask2 = np.zeros_like(edges)

    ignore_mask_color = 255   
    # # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(-300,imshape[0]),(400,200), (750, 200), (imshape[1]+300,imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask1, vertices, ignore_mask_color)

    vertices2 = np.array([[(75,imshape[0]),(320,250), (700, 250), (1050,imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask2, vertices2, ignore_mask_color)

    mask2 = cv2.bitwise_not(mask2)
    mask = cv2.bitwise_not(mask1+mask2)
    
    #plt.imshow(mask)
    
    masked_edges = cv2.bitwise_and(edges, mask)#masked_edges=edges+mask
    

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 3     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 6 #minimum number of pixels making up a line
    max_line_gap = 1.5    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    SloperightSum = 0  #weighted sum
    YinterceptrightSum = 0 #weighted sum
    SlopeleftSum = 0
    YinterceptleftSum = 0 

    Numright = 0  #weight
    Numleft = 0
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    
    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    #plt.imshow(line_image)
    
    # Iterate over the output "lines" and draw lines on a blank image
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             if 0.7>((y2-y1)/(x2-x1))>0.2 :
#                 SloperightSum += (((y2-y1)/(x2-x1))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2))
#                 YinterceptrightSum += (y1-((y2-y1)/(x2-x1))*x1)*np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
#                 Numright += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

#             if -0.8<((y2-y1)/(x2-x1))<-0.2 :
#                 SlopeleftSum += (((y2-y1)/(x2-x1))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2))
#                 YinterceptleftSum += (y2-((y2-y1)/(x2-x1))*x2)*np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
#                 Numleft += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


#     # slope values k and y intercept values b
#     kl = SlopeleftSum/Numleft
#     bl = YinterceptleftSum/Numleft

#     kr = SloperightSum/Numright
#     br = YinterceptrightSum/Numright

#     # x coordinates for interception points at top and bottom of the mask
#     Topxl = math.floor((320-bl)/kl)
#     Topxr = math.floor((320-br)/kr)
#     Botxl = math.floor((540-bl)/kl)
#     Botxr = math.floor((540-br)/kr)

#     # plot left and right lane lines
#     cv2.line(line_image,(Topxl,320),(Botxl,540),(255,0,0),10)
#     cv2.line(line_image,(Topxr,320),(Botxr,540),(255,0,0),10)
   

    # Create a "color" binary image to combine with line image
    imagefromeye = np.copy(image)


    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(imagefromeye, 0.8, line_image, 1, 0) 
    #plt.imshow(lines_edges)


        # you should return the final output (image where lines are drawn on lanes)

    return lines_edges


# In[19]:

white_output = 'Knights_clip_output.mp4'
clip1 = VideoFileClip("Knights_clip.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# In[20]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

