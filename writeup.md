# Advanced Lane Detection - 

### <p style="color:gray">_Udacity - Self driving Car - Project 2_</p>

### Techniques used for Lane Detection in this project
1. Calibration of Camera - for precision
2. Perspective Transformation - changing the Front view of the lanes to Top view 
3. Use of Color and Sobel Filters - for identification of Yellow and White color lane edges
4. Use of Histogram, Sliding Window and Polynomial function - to fit a polynomial on the data from filters

### High-Level Steps
* Compute the camera **calibration matrix and distortion coefficients** using a set of given chessboard images
* After Camera Calibration - 
* Rest of the steps apply for all test-images/frames<br>
  1. Correct distortion of the image/frame using the calibration data from step-1<br>
  2. Transform perspective of the image i.e change from front view to top view<br>
  3. Use color and Sobel filters to detect the edges and change the data to binary (1 for all detected edges)<br>
  4. Calculate histogram using binary image data and divide the windows in n-regions<br>
  5. Identify the peaks of histogram and define a small window to identify pixels close to the peak edges<br>
  6. Fit polynomial using the pixels data - On this case the fit for for X coordinates (X = F(Y))<br>
  7. The polynomial fitted are the lanes idenified<br>
  8. Calculate metrics - Radius of curvature and offset of car center from the lane center<br>
  9. Transform perspective back to original - change top view to original front view<br>
  10. Present expected output - Fill the idenified lanes with color and display, ROC and Car center offset<br>
  

[//]: # (Image References)

[image0]: ./output_images/calibration_output.png "Calibration Output"
[image1]: ./output_images/straight_lines2_Original-Distorted.jpg "Original Distorted"
[image2]: ./output_images/straight_lines2_Undistorted.jpg "Undistorted"
[image3]: ./output_images/straight_lines2_4Src_Points.png "4 Source Points"
[image4]: ./output_images/straight_lines2_Warped.jpg "Top View of Lanes"
[image5]: ./output_images/all_binaries.png "Color and Sobel Filter individual outputs"
[image6]: ./output_images/straight_lines2_Combined.jpg "Filtered Edges with Binary data"
[image7]: ./output_images/straight_lines2_Histogram_2.png "Histogram of Filtered edges"
[image8]: ./output_images/straight_lines2_PolyWin.jpg "Sliding Window"
[image9]: ./output_images/straight_lines2_PolyFit.jpg "Fit line using Polynomial function"
[image10]: ./output_images/ROC_Formula.png "ROC Formula"
[image11]: ./output_images/straight_lines2_Final-Image.jpg "Final Detected Lanes"
[video1]: ./test_videos_output/project_video.mp4 "Video"

# <p style="color:blue" > Code Steps and Details </p>

## Code Planning
* I have attempted to modularize and parametrize the code as much as possible to reduce complexity and make it easy to tune hyperparameters
* I have used 3 Classes:
  1. **Class HyperParameter** : is used to maintain hyperparameters used by functions. This makes it easy to tune the        parameters on the fly without any chane in the code

  2. **Class Save_Images** : is used to save all the images generated, initially in this dictionary and then in a            pickle file. Saving it in pickle file gives advanatage to view the o/p images for analysis/debugging without          rerunning the program 

  3. **Class Line** : 
       This class is used to maintain information about lane detection and curvature. This information can be used to        average the curvature points to determine the next starting points for lane detection

* The code is divided in:
  1. Helper Functions - Modularize repeated code
  2. Main Functions  - These functions form the main Pipeline
  3. Testing Function - These functions are use to run the pipeline on Test images
  4. Production Function - These function forms the main Pipeline to run against the real video
  5. Visualization - Save Images in Dictionry and then to Pickle files for later debugging and analysis

## Step-1-A:  Camera Calibration

calibration steps:
1. Identify 15-20 images for calibration- Here we use chessboard images
2. Decide the number of corners of the chessboard per row (nx=9) and per column (ny=6) to be identified
3. Create object points using np.mgrid
4. Read each calibration image and use **cv2.findChessboardCorners** to detect the required (nx, ny) corners
5. These corners will form on set of Image points - 
6. For every image, collect the image points and make copy of the object points in two seperate list
7. Pass this list of image points and object points to **cv2.calibrateCamera**
8. **cv2.calibrateCamera** then returns the Camera Matrix and Distortion cooefficients which will be then used to undistort any image

Notes:
1. **cv2.drawChessboardCorners** CV function has no direct contribution to calibration but it helps visually to determine if the detected corners are correct or no
2. For Images which do not have the needed (nx, ny) corners, **cv2.findChessboardCorners** will not return success.
3. Those images are good to test how good the calibration is performed on the iamges by the calibration Matrix.

_camera calibration function is called from the initialize env function defined in next cell_


```
def detect_and_draw_corners(path, paramInstance, key,  calibImagesInstance):
    paramDict = paramInstance.get_paramDict(key)
    nx = paramDict["NX_CORNERS_PER_ROW"]
    ny = paramDict["NY_CORNERS_PER_COLUMN"]
    
    objp = np.zeros((nx * ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    imgPoints = []
    objPoints = []
    
    for filename in glob.glob(path + "*.jpg"):
        originalImage = mpimg.imread(filename)
        calibImagesInstance.save_images_dict("CALIB_IMAGES", "ORIGINAL", [filename,originalImage])
        distortedImage = np.copy(originalImage)
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_RGB2GRAY)
        imgSize = grayImage.shape[::-1]
        ret, corners = cv2.findChessboardCorners(grayImage, (nx, ny), None)
        if ret:
            winSize  = paramDict["WINSIZE"]
            zeroZone = paramDict["ZEROZONE"]
            criteria = paramDict["CRITERIA"]
            corners  = cv2.cornerSubPix( grayImage, corners, winSize, zeroZone, criteria );
            cv2.drawChessboardCorners(distortedImage, (nx, ny), corners, ret)
            imgPoints.append(corners)
            objPoints.append(objp)
            calibImagesInstance.save_images_dict("CALIB_IMAGES", "WITHCORNERS",[filename,distortedImage])
           
    return imgSize, imgPoints, objPoints
            
               
def camera_calibration(path, paramInstance, key, calibImagesInstance):
    """
    returnedTuple = (success, cameraMatrix, distortionCoeff, rvecs, tvecs)
    """
    paramDict = paramInstance.get_paramDict(key)
    imgSize, imgPoints, objPoints = detect_and_draw_corners(path, paramInstance, "DETECT_AND_DRAW_CORNERS", calibImagesInstance)
    returnedTuple =  cv2.calibrateCamera(objPoints, imgPoints, imgSize, None, None)
    
    success = returnedTuple[0]
    cameraMatrix = returnedTuple[1]
    distortionCoeff = returnedTuple[2]
 
    if success:
        updateDict = {"CAMERA_CALIBRATION" : {"CAMERAMATRIX" : cameraMatrix,
                                              "DISTORTIONCOEFF" : distortionCoeff
                                            }
                     }
        pickledump_calib_images(calibImagesInstance)
        paramInstance.set_paramDict(updateDict)
    else:
        print("Something wrong with Calibration!!")
        raise
        
    return cameraMatrix, distortionCoeff
```

Below is output of Camera Calibration 

![alt text][image0]


# Step-1-B: Initialize the environment

**initialize_env**
This is the first function called in the pipeline. We create instances of the 3 classes: HyperParameter, Line and the Save_images classes. We then generate the Camera Matrix and Distortion coeeficients. The calibration data is either read from the existing pickle file **calibration_data.pickle**, if it exists or it is generated by calling the **camera_calibration**. **calibration_data.pickle** is created to avoid calling the **camera_calibration** function for every frame.

The 3 classes instances and the Calibration data once set from this function will act as Global variables for the rest of the functions

```
def initialize_env():
    paramInstance = HyperParameters()
    lineInstance = Line()
    testImagesInstance = Save_Images()

    dirDict = paramInstance.get_paramDict("PATH")
    fileDict = paramInstance.get_paramDict("PICKLE_FILES")
    calibPickleFile = dirDict["PICKLE"] + fileDict["CALIB"] 
    calibImagesPath = dirDict["CALIB"]
    if (not os.path.exists(calibPickleFile)):
        cameraMatrix, distortionCoeff = camera_calibration(calibImagesPath, paramInstance, "CAMERA_CALIBRATION", testImagesInstance)
        updateDict = {"CAMERA_CALIBRATION" : {"CAMERAMATRIX" : cameraMatrix,
                                              "DISTORTIONCOEFF" : distortionCoeff
                                            }
                     }
        pickle_dump(calibPickleFile, updateDict)
        paramInstance.set_paramDict(updateDict)
    else:
        cameraMatrix, distortionCoeff = unpickle_calibration_data(calibPickleFile)
        updateDict = {"CAMERA_CALIBRATION" : {"CAMERAMATRIX" : cameraMatrix,
                                              "DISTORTIONCOEFF" : distortionCoeff
                                            }
                     }
        paramInstance.set_paramDict(updateDict)
    return paramInstance, lineInstance, testImagesInstance, cameraMatrix, distortionCoeff
```

For rest of the steps I will be using ouput of the test_image straight_lines2.jpg

Lets take a look first at the Original Undistorted Test image 

![alt text][image1]


## Step-2: Undistort Image
**undistort_image :** This function uses the Calibration data (_cameraMatrix, distortionCoeff_) generated in Step-1 to undistort the image/frame
```
def undistort_image(image, cameraMatrix, distortionCoeff):
    return cv2.undistort(image, cameraMatrix, distortionCoeff, None, cameraMatrix)
```
After running this step we get Undistorted image as below

![alt text][image2]


## Step-3: Perspective Transformation:
For perspective tranformation I have created two functions:

**tranformation_matrices:** 
Generates the transformation and Inverse transformation Matrices - Need to be called only once for all images

**topview_perspective_transform:** 
It is the main pipeline function which applies transformation to the undistorted image. It uses the **M** matrix generated by transformation_matrices function. M and Minv matrices need to be generated only once ideally.

1. Collect 4 Source and 4 Destination points
   To get the source points, 
   a. Pick one of the straight lane images (straight_lines1.jpg or straight_lines2.jpg)
   b. Pick 4 points on exactly on the lanes (like 4 corners of trapeziod) (use a paint or some app)
   c. Try to match the Y co-ordinates of 2 top and 2 bottom points
   For the Destination points I kept the height same and took little offset from the width
   
2. Feed those source and destination points to cv function **cv2.getPerspectiveTransform**. Based on the position of the source points and destination points we can get either the **M** Transformation matrix or **Minv** Inverse transformation matrix
     M = cv2.getPerspectiveTransform(**src**, dst)      -- Transformation matrix
     MInv = cv2.getPerspectiveTransform(dst, **src**)   -- Inverse Transformation Matrix
     
3. Once we get the M and Minv matrix, this will be applied to all the images during perspective transformation step.

Below image shows how 4 source points were selected

![alt text][image3]

_code snippet:_
```
def tranformation_matrices(imgSize, paramInstance, key):
    """
    This function needs to be evaluated once
    It is assumed that the imgSize will remain constant for all Test Images and VideoFrames
    """
    paramDict = paramInstance.get_paramDict(key)
    left= paramDict["LEFT_BOTTOM_CORNER"]
    right=paramDict["RIGHT__BOTTOM_CORNER"] 
    apex_left=paramDict["LEFT_TOP_CORNER"]
    apex_right=paramDict["RIGHT_TOP_CORNER"] 

    height, width = imgSize
    X_offset = width//5
    Y_offset = 0
    
    src=np.float32([left,apex_left,apex_right,right]) 
    dst = np.float32([[X_offset, height], [X_offset, Y_offset], [width - X_offset, Y_offset],[width - X_offset, height]])
    M = cv2.getPerspectiveTransform(src, dst)
    MInv = cv2.getPerspectiveTransform(dst, src)
    return M, MInv


def topview_perspective_transform(image, M, warpImgSize, paramInstance, key): 
    """
    Create top view of the lanes
    """
    warpedImage = cv2.warpPerspective(image, M, warpImgSize, flags=cv2.INTER_LINEAR)
    return warpedImage

```

Below image is result of perspective transformation (Tranformed into Top View)

![alt text][image4]


## Step-4: Apply Color and Gradient Filter to detect White and Yellow Lane Lines
**gradient_by_color_and_filter :** This is the main function called for this step. Rest listed are helper functions.

This step requires more exploration and analysis. We need to try various thresholds for the colors and Sobel filters along with different kernel size. There are many different color types, I explored, RGB, HSV and HLS.
After analyzing RGB, HSV and HLS colors, Red, Green and Saturation channel were used to detect the yellow and white color lane edges
On the gradient side I used Sobel Filter to generate three binaries : Absolute Gradient in X-direction, Magnitude in X-Y direction and the Direction of the edges

steps:
1. Get the R, and B channels from RGB image. apply threshold and convert data to Binary. 1's for Edges detected.
2. Get the S channel from HSL image. apply threshold and convert data to Binary. 1's for Edges detected.
3. Get the Sobel filters Absolute, magnitude and Direction. Apply threshold and convert data to Binary. 1's for Edges detected.
4. Finally combine the output of all the filters with binary data. 

_The images are converted to binary (1's and 0's) so that is is easy to identify histogram peaks which will be only where 1's are._

_code snipet_

```
def get_channel_binary(channel_img, paramDict, imgType="RGB", channel_color='R'):
    prefix = imgType + "_" + channel_color
    thresh = (paramDict[prefix + "_MIN_THRESHOLD"], paramDict[prefix + "_MAX_THRESHOLD"])
    
    channel_binary = np.zeros_like(channel_img)
    channel_binary[(channel_img >= thresh[0]) & (channel_img <= thresh[1])] = 1
    
    return channel_binary

def get_color_channel_binaries(img, paramDict):
    """
    img is original Warped color image - 
    by default it is RGB since we used mpimg.imread to read the original image
    For project, will be using only R, G and S channels for detection of lane lines
    As they give better results for White and Yellow lanes
    """
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    R, G, B  = (img[:,:,0], img[:,:,1], img[:,:,2])  # using B only for Analysis
    H, L, S  = (HLS[:,:,0], HLS[:,:,1], HLS[:,:,2])
    R_binary = get_channel_binary(R, paramDict, imgType="RGB",channel_color='R')
    G_binary = get_channel_binary(G, paramDict, imgType="RGB",channel_color='G')
    S_binary = get_channel_binary(S, paramDict, imgType="HLS",channel_color='S')
    
    return R, G, B, H, L, S, R_binary, G_binary, S_binary

def get_sobel_binary(gray, paramDict, orient='X', flag='ABS'):
    if flag == "ABS":
        flag += orient
    
    sobel_kernel = paramDict["SOBEL_KERNEL"]
    
    thresh = (paramDict["SOBEL_"+flag+"_MIN_THRESHOLD"], paramDict["SOBEL_"+flag+"_MAX_THRESHOLD"])
    
    gradientx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    gradienty = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    if (flag == "ABSX"):
        value = np.absolute(gradientx)
    elif (flag == "ABSY"):
        value = np.absolute(gradienty)
    elif flag == "MAG":
        value = np.sqrt((gradientx ** 2) + (gradienty ** 2))
    elif flag == "DIR":
        value = np.arctan2(np.absolute(gradienty), np.absolute(gradientx))
        
    if flag == "DIR" :
        scaled_sobel = value
    else:
        scaled_sobel = np.uint8(255*value / np.max(value))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(value >= thresh[0]) & (value <= thresh[1])] = 1
    
    return binary_output

def gradient_by_color_and_filter(fname, warpedImage, paramInstance, key, testImagesInstance, mode): 
    paramDict = paramInstance.get_paramDict(key)
    gray = cv2.cvtColor(warpedImage, cv2.COLOR_RGB2GRAY)
    grayBlur = gaussian_blur(gray, paramInstance, "GAUSSIAN_BLUR")
    
    RGSImages = get_color_channel_binaries(warpedImage, paramDict)
    R, G, B, H, L, S, R_binary, G_binary, S_binary = RGSImages
    sobelx_binary = get_sobel_binary(grayBlur,  paramDict, orient='X', flag = 'ABS')
    mag_binary = get_sobel_binary(grayBlur, paramDict, orient='X', flag = 'MAG')
    dir_binary = get_sobel_binary(grayBlur, paramDict, orient='X', flag = 'DIR')
    
    combined_binary = np.zeros_like(sobelx_binary)
    combined_binary[(sobelx_binary == 1 ) | ((mag_binary == 1) & (dir_binary == 1)) |
                    (R_binary == 1) | (G_binary == 1)  | (S_binary == 1)] = 1
    
    if mode == "TESTRUN":
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["GRAY"] = gray
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["GRAY_BLUR"] = grayBlur
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["R_CHANNEL"] = R
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["G_CHANNEL"] = G
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["B_CHANNEL"] = B
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["H_CHANNEL"] = H
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["L_CHANNEL"] = L
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["S_CHANNEL"] = S  
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["R_BINARY"] = R_binary
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["G_BINARY"] = G_binary
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["S_BINARY"] = S_binary
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["SOBEL_ABSX"] = sobelx_binary
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["SOBEL_MAG"] = mag_binary
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["SOBEL_DIR"] = dir_binary
       
    return (combined_binary)
```

Below is Output of all channels 


![alt text][image5]


Below is output of the combined image


![alt text][image6]




## Step-5: Find Lane Lines using Sliding Window and Fit Polynomial 
**initial_fit_polynomial :** This function calls the sliding window technique to detect the pixels within a small defined windows around the left and right edges detected in the Gradient Step: A histogram of the gradient values is used to identify maximum change and infer them as Lane Lines
The sliding window is not very efficient to do for all frames. "initial" is used in the function name as to indicate that this function is necessary atleast for the first step.

**find_lane_pixels :** Helper function called by **initial_fit_polynomial** to generate the histogram and identify the pixels close to the right and left lanes within given margin

Once the left and right lane pixels are generates (X data points) we use this to fit polynomial
X = F(Y) = A\*Y^2 + B\*Y + C

steps:
1. Create Histogram and identify beggining central pixels for of left and right lanes for the sliding windows
   a. Read the filtered Edges binary image output from the filteration step. This image id made up of 1's and 0's
   b. Create a histogram data by summing up all the columns(1280). The peak of the edges will have maximum value.
   c. Divide the image from the center (in the width) into Left and right parts
   d. Identify Two peaks, one in the left region and another in the right region
   e. Those peaks will be the center of the first sliding window starting at the bottom
2. Create the Sliding window
   a. Define size for the sliding window
   b. Using the centers identified above find diagnonal points of the window for both left and right lanes
   c. Collect the pixels points (with values 1 i.e ones part of edges) within the window
   d. If the number of pixels are greater than defined minpixels threshold, then recenter the window
   e. Collect the (x,y) coordinates of all the poxels and store it in Lists, one each for left and right lanes
   f. Repeat this for all the sliding windows until we get the pixels for the full lane from bottom to top
   g. For easy identification we have colored those pixel points. Red for left lane and Blue for right lane
   h. For visualization we also draw those rectangle windows on the image as shown below
   
3. Fit <u>2nd order Polynomial:</u> **X = F(Y) = A*Y^2 +B*Y + C** to the pixel points indeitified using the sliding window above
   a. Using the pixel points collected using the sliding window we fit a polynomial to draw the lane curve
   b. we use 

The code snipet below shows only the main functions and not the helper function

_Code Snipet_  


```
def find_lane_pixels(fname, binary_warped, paramDict,ploty, mode, testImageInstance ):
    """
    pixelPositions: leftx  : X-Coordinates of LeftLane,  along high histogram peak, of All defined Sliding Windows
                    lefty  : Y-Coordinates of LeftLane,  along high histogram peak, of All defined Sliding Windows
                    rightx : X-Coordinates of RightLane, along high histogram peak, of All defined Sliding Windows
                    righty : Y-Coordinates of RightLane, along high histogram peak, of All defined Sliding Windows
    """
    nwindows = paramDict["NWINDOWS"]
    margin   = paramDict["MARGIN"]
    minpix   = paramDict["MINPIX"]
    
    histogram      = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img        = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint       = np.int(histogram.shape[0]//2)
    leftx_base     = np.argmax(histogram[:midpoint])
    rightx_base    = np.argmax(histogram[midpoint:]) + midpoint
    leftx_current  = leftx_base
    rightx_current = rightx_base
    imageHeight    = binary_warped.shape[0]
    windowHeight   = np.int(binary_warped.shape[0]//nwindows)
    nonZero        = binary_warped.nonzero()     # [(y1, y2,...)(x1, x2, ...)]. All Indices where values are 1
    nonZeroY       = np.array(nonZero[0])       # Just Y indices (y1, y2, y3...)  ..values range from 0..720
    nonZeroX       = np.array(nonZero[1])       # Just X indices (x1, x2, x3...)  ..values range from 0..1280
    left_lane_inds = []
    right_lane_inds = []
    staticData = (imageHeight, windowHeight, margin )
    for n in range(nwindows):
        leftDiagonalPts, rightDiagonalPoints = rectangle_diagonal_points(n, staticData, leftx_current, rightx_current ) 
        if mode == "TESTRUN":
            cv2.rectangle(out_img,leftDiagonalPts[0],leftDiagonalPts[1],(0,255,0), 4) 
            cv2.rectangle(out_img,rightDiagonalPoints[0],rightDiagonalPoints[1],(0,255,0), 4)
           
        good_left_inds = nonZeroPixelsInWindow(nonZeroX, nonZeroY, leftDiagonalPts)
        good_right_inds = nonZeroPixelsInWindow(nonZeroX, nonZeroY, rightDiagonalPoints)
 
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # recenter next window on their mean position
        leftx_current = recenter_window(leftx_current, nonZeroX, good_left_inds, minpix)
        rightx_current = recenter_window(rightx_current, nonZeroX, good_right_inds, minpix)
        
    try:
        left_lane_inds = np.concatenate(left_lane_inds)   #Flatten -- [[1,2,3, [4,5,6]]] => [1,2,3,4,5,6]
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        print("value error in find_lane_pixes module!!!!")
    
    # Extract left and right line pixel positions
    pixelPositions = extract_pixel_positions(nonZeroX, nonZeroY ,left_lane_inds, right_lane_inds )
    polyfitData = get_polyfit_data(pixelPositions, ploty)
    left_fitx, right_fitx = (polyfitData[2], polyfitData[3])
    
    out_img[nonZeroY[left_lane_inds], nonZeroX[left_lane_inds]] = [255, 0, 0]
    out_img[nonZeroY[right_lane_inds], nonZeroX[right_lane_inds]] = [0, 0, 255]
    
    polyWinData = [out_img, left_fitx, right_fitx, ploty]
    visual_data = [pixelPositions, polyfitData,nonZeroY, nonZeroX, left_lane_inds, right_lane_inds, margin, ploty]
    
    if mode == "TESTRUN":
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["POLYWIN_IMGS"] = polyWinData
        testImagesInstance.SAVED_IMAGES_DICT["TEST_IMAGES"][fname]["HISTOGRAM"] = histogram
    
    return  out_img, histogram, visual_data
 
def initial_fit_polynomial(fname, binary_warped, ploty, paramInstance, key, testImagesInstance, mode):
    """
    polyfitData: Derived using pixelPositions data and polyfit function
               : left_fit   : Coeffs (A,B,C) of the Left-Fit Polynomial (AY^2 + BY + C)
               : right_fit  : Coeffs (A,B,C) of the Right-Fit Polynomial (AY^2 + BY + C)
               : left_fitx  : LeftLane X-values derived using polyFunc (AY^2 + BY + C) for each Y in 0-710(imgHeight)
               : right_fitx : RightLane X-values derived using polyFunc (AY^2 + BY + C) for each Y in 0-710(imgHeight)
    """
    paramDict = paramInstance.get_paramDict(key)
    out_img_1, histogram, visual_data = find_lane_pixels(fname, binary_warped, paramDict,ploty, mode, testImagesInstance)
    out_img_2 = create_polyfill_outimg_for_visual( fname, binary_warped, visual_data, testImagesInstance, mode)
    pixelPositions, polyfitData = (visual_data[0], visual_data[1])
    return polyfitData, pixelPositions, histogram, out_img_1, out_img_2   #(left_fit, right_fit, left_fitx, right_fitx) 

```
Below image shows the Histogram generated


![alt text][image7]

   
Below image shows is generated from this step to show how the sliding window works
 
 
![alt text][image8] 



Below image shows region is defined to fit the polynomial curve

![alt text][image9] 




# Step-6: Calculate Curvature Data 

### **measure_curvature_real**
In this function, we calculate two metrics:
1. ROC - Radius of Curvature of both left and right Lane is calculated as shown below. 
_credits_ : Udacity course : Self Driving Car Engineer :


![alt text][image10] 





2. Offset of Car center from Lane Center:
Ideally the center of the car is expected to be the center of the lane i.e it is assumed that the car is driving exactly in the center of the lane and since it is also assumed that the camera is placed in the center of the car, say on the dashboard. Practically the car may not be running in exactly center of the lanes so we calculate here what is that offset i.e how much left or right is the center of the car from the center of the lane:
1. Center of the lane is : (Width of image)/2
2. Since Max of Y-coordinate (bottom of the image) will be the point closest to the Car, we evaluate poynomial    
   function to get the respective X coordinates for the left and right lanes. and then we calculate the absolute  
   difference of those X positions
   actualCarCenter= (leftX + rightX)/2    

_code Snipet_
```
def measure_curvature_real(ploty, xWidth, fitData,  paramInstance, key):
    '''
    Calculates the curvature of polynomial functions in meters.
    Radius of curvature is defined at maximum Y-value - i.e bottom of the image
    '''
    paramDict = paramInstance.get_paramDict(key)
    ym_per_pix = paramDict["Y_METERS_PER_PIXEL"] # meters per pixel in y dimension
    xm_per_pix = paramDict["X_METERS_PER_PIXEL"] # meters per pixel in x dimension
    
    leftFit, rightFit = fitData
    y_max = np.max(ploty)
    Y_meters = y_max * ym_per_pix
    leftRoc = evaluate_polynomial_func(leftFit, Y_meters, 1) 
    rightRoc = evaluate_polynomial_func(rightFit, Y_meters, 1)
    avgROC = (leftRoc + rightRoc)/2
    
    idealCarCenter = xWidth/2
    leftX =  evaluate_polynomial_func(leftFit, Y_meters, 2) 
    rightX = evaluate_polynomial_func(rightFit, Y_meters, 2) 
    actualCarCenter= (leftX + rightX)/2
    distanceFromLaneCenter = (idealCarCenter - actualCarCenter) * xm_per_pix
    
    return leftRoc, rightRoc, avgROC,  distanceFromLaneCenter
```

## Step-6: Draw final detected Lane Lines and fill it with color as it will be viewed by autonomous car

So far we have detected the lanes using the top view of the image and its filtered binary form.
It is now time to project back the identified lanes back on the original undistorted image
For this I have function **draw_final_lanes** : 
In this function we achieve three things
    1. Inverse transform the Warped Image- back to original image using **Minv**
    2. We draw a polygon around the detected lanes and fill it with green color. 
    3. In addition we display the Avg ROC and car distance from lane center on the image
    
_code snipet_

```
def draw_final_lanes(undistortedImage, combinedImage, imgSize, Minv, ploty, fitXData, curvatureData, paramInstance, key):
    paramDict = paramInstance.get_paramDict(key)
    orgImgWeight  = paramDict["ORGIMG_WEIGHT"]
    warpImgWeight = paramDict["WARPIMG_WEIGHT"]
    left_fitx , right_fitx = fitXData
    avgRoc, distanceFromLaneCenter = (curvatureData[2],curvatureData[3])
    ROCText = "Avg Radius of curvature is: {:.2f}".format(avgRoc) + " m"
    distanceText = "Car is: {:.2f}".format(np.abs(distanceFromLaneCenter)) 
    
    if distanceFromLaneCenter < 0:
        distanceText += " m Left from Lane Center "
    else:
        distanceText += " m Right from Lane Center"
    
    warp_zero = np.zeros_like(combinedImage).astype(np.uint8)     
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])   
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    imgsize = (undistortedImage.shape[1], undistortedImage.shape[0] )
    newwarp = cv2.warpPerspective(color_warp, Minv, imgsize)
       
    result = cv2.addWeighted(undistortedImage, orgImgWeight, newwarp, warpImgWeight, 0)
    
    cv2.putText(result,ROCText, (50,50), 2, 1, (255,255,255),2)
    cv2.putText(result,distanceText, (50,100), 2, 1, (255,255,255),2)
    
    return result

```

Below is the final image generated and this is how it will be seen during the lane detection in the video


![alt text][image11]



Once the pipeline is tested thoroughly, we take the code to run against the provided movie clip



Here's a [link to my video result](./test_videos_output/project_video.mp4)



# Do we stop here? No

## Issues encountered and few thoughts on how we can improve

Even though this is advanced lane detection, there is still lot more to be done. There is room for optimization or better performance. Here are few things we can consider

Issues observed:
1. The code depends on manually measured source pointswhich are used to generate the transformation Matrices for perspective transformation

2. The code encounters a very short flicker around when the illumination is very high and when there is sudden change of illumination (like a bumper)

3. The code failed for the challenged videos which has more curves and more illumination changes

How we can improve the pipeline
1. Smoothing technique which can be achieved as follows<br>
   a. Track all the fit line data (fit pixels and co-efficients) for every frame<br>
   b. Save last 5 frames data<br>
   c. cmaintain average of the polynomial fit coefficients.<br>
   d. Use this average for determine the center pixels for the next frame<br>
   e. We will also maintain a sanity check if the lanes were detected correctly by checking the distances between           The lanes at few different points of the detected lane.<br>
   g. If the pipeline fails to detect lane correcty for say 10 times then we can use the sliding window technique again.