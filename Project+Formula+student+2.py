
# coding: utf-8

# In[168]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
get_ipython().magic('matplotlib inline')


# In[169]:


#reading in an image and turn it into grayscale
image = mpimg.imread('FSnew.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(gray)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# In[170]:

#Gaussian Blur and Canny!
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
plt.imshow(edges)


# In[171]:

# Next we'll create a masked edges image using cv2.fillPoly()
mask1 = np.zeros_like(edges)
mask2 = np.zeros_like(edges)

ignore_mask_color = 255   
# # This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(-600,imshape[0]),(800,400), (1500, 400), (imshape[1]+600,imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask1, vertices, ignore_mask_color)

vertices2 = np.array([[(150,imshape[0]),(850,550), (1300, 550), (2100,imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask2, vertices2, ignore_mask_color)

mask2 = cv2.bitwise_not(mask2)
mask = cv2.bitwise_not(mask1+mask2)

masked_edges = cv2.bitwise_and(edges, mask)#masked_edges=edges+mask
# mask2 = np.ones_like(edges)
# mask2 = cv2.bitwise_not(mask2)
# mask_color2 = 0
# vertices2 = np.array([[(250,imshape[0]),(750,650), (1300, 600), (1800,imshape[0])]], dtype=np.int32)
# cv2.fillPoly(mask2, vertices, ignore_mask_color)
# print (mask2[1000][600])
#masked_edges2 = cv2.bitwise_and(masked_edges, mask2)#masked_edges=edges+mask
plt.imshow(masked_edges)
# plt.imshow(mask)


# In[172]:

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 1     # minimum number of votes (intersections in Hough grid cell)
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
plt.imshow(line_image)

# Convert Houghlines into the form y=kx+b, use line length as weights, take weighted average of k and b.
# for line in lines:
#     for x1,y1,x2,y2 in line:
#         if 0.8>((y2-y1)/(x2-x1))>0.2 :
#             SloperightSum += (((y2-y1)/(x2-x1))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2))
#             YinterceptrightSum += (y1-((y2-y1)/(x2-x1))*x1)*np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
#             Numright += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

#         if -0.8<((y2-y1)/(x2-x1))<-0.2 :
#             SlopeleftSum += (((y2-y1)/(x2-x1))*np.sqrt((x1 - x2)**2 + (y1 - y2)**2))
#             YinterceptleftSum += (y2-((y2-y1)/(x2-x1))*x2)*np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
#             Numleft += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            
# # slope values k and y intercept values b
# kl = SlopeleftSum/Numleft
# bl = YinterceptleftSum/Numleft

# kr = SloperightSum/Numright
# br = YinterceptrightSum/Numright

# # x coordinates for interception points at top and bottom of the mask
# Topxl = math.floor((320-bl)/kl)
# Topxr = math.floor((320-br)/kr)
# Botxl = math.floor((540-bl)/kl)
# Botxr = math.floor((540-br)/kr)

# # plot left and right lane lines
# cv2.line(line_image,(Topxl,320),(Botxl,540),(255,0,0),10)
# cv2.line(line_image,(Topxr,320),(Botxr,540),(255,0,0),10)





# In[173]:

# Create a "color" binary image to combine with line image
imagefromeye = np.copy(image)
# Draw the lines on the edge image
lines_edges = cv2.addWeighted(imagefromeye, 0.8, line_image, 1, 0) 
plt.imshow(lines_edges)


# In[ ]:




# In[ ]:



