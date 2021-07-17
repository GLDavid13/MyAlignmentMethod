import cv2 as cv
from astropy.io import fits
import numpy as np

class MyAlignment:
    """
    Summary
    -------
    This is the main class as I wanted it to be simple.
    Don't start to blame me, I don't pretend to be a Python super-star.
    This project, after all, was just for entertainment and if it can be reused 
    elsewhere, you are welcome by advance.
    Coming back to this class, it contains everything you need to run an 
    alignment using OpenCV, astropy and numpy as main libraries.
    Have fun!

    Methods
    -------
    loadImage(imgPath)
        Load any king of image (but not fit image!)
    
    loadImage(imgPath, isFit=False)
        Overloaded of previous one but with a boolean in case you want to load 
        a fit image

    loadFitImage(cls, fitPath)
        Class method used to load a fit image
    
    fitAsOpenCVImage(cls, fit)
        Class method to convert any fit image into OpenCV one

    saveImage(img, absolutePath)
        Save any OpenCV image into your wanted format and your given absolute 
        path

    sobel(img, ddepth, scale, delta)
        The Sobel algorithm for contour using any OpenCV image and for a given 
        depth, scale and delta

    contour(cls, img)
        Class method, this is the OpenCV call to findContour in case we want to 
        use it instead of sobel

    drawContours(cls, img, contours)
        Class method used to convert an OpenCV contour into an OpenCV image

    align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False)
        This is the main method to align an image to a template one
    """
    
    def loadImage(imgPath):
        """
        Summary
        -------
        Load a bitmap image and returns its OpenCV related object

        Args:
            imgPath (string): Absolute path of bitmap image to load

        Returns:
            Any: OpenCV object as image representation
        """
        return cv.imread(imgPath)

    def loadImage(imgPath, isFit=False):
        """
        Summary
        -------
        Load a bitmap image and returns its OpenCV related object

        Args:
            imgPath (string): Absolute path of bitmap image to load
            isFit (bool, optional): Set if image is fit format. 
            Defaults to False.

        Returns:
            Any: OpenCV object as image representation
        """
        if(isFit):
            return MyAlignment.fitAsOpenCVImage(
                MyAlignment.loadFitImage(imgPath)
            )
        return cv.imread(imgPath)

    @classmethod
    def loadFitImage(cls, fitPath):
        """
        Summary
        -------
        Class method to load a fit image and returns its astropy related object

        Args:
            fitPath (string): Absolute path of fit image to load

        Returns:
            HDUList: astropy object as image representation
        """
        return fits.open(fitPath)

    @classmethod
    def fitAsOpenCVImage(cls, fit):
        """
        Summary
        -------
        Convert an astropy fit image into an OpenCV representation

        Args:
            fit (HDUList): astropy fit image

        Returns:
            Any: OpenCV image representation
        """
        img = fit[0].data
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        return img

    def saveImage(img, absolutePath):
        """
        Summary
        -------
        Save an OpenCV image object as bitmap file

        Args:
            img (Any): OpenCV image object
            absolutePath (string): Absolute file path where to save
        """
        cv.imwrite(absolutePath, img)

    def sobel(img, ddepth, scale, delta):
        """
        Summary
        -------
        Sobel algorithm for contours

        Args:
            img (Any): OpenCV image representation
            ddepth (int): output image depth, see filter_depths in OpenCV
            "combinations"; in the case of 8-bit input images it will result 
            in truncated derivatives
            scale (int): scale factor for the computed derivative values; 
            by default, no scaling is applied
            delta (float): [delta value that is added to the results prior 
            to storing them in dst

        Returns:
            Any: OpenCV image of Sobel contours
        """
        src = cv.GaussianBlur(img, (3, 3), 0)
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(
            gray, 
            ddepth, 
            1, 
            0, 
            ksize=3, 
            scale=scale, 
            delta=delta, 
            borderType=cv.BORDER_DEFAULT
        )
        grad_y = cv.Sobel(
            gray, 
            ddepth, 
            0, 
            1, 
            ksize=3, 
            scale=scale, 
            delta=delta, 
            borderType=cv.BORDER_DEFAULT
        )
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad

    @classmethod
    def contour(cls, img):
        """
        Summary
        -------
        Get contours of an image using OpenCV findContours function

        Args:
            img (Any): OpenCV image representation

        Returns:
            Any: OpenCV image with contours only
        """
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        return cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    @classmethod
    def drawContours(cls, img, contours):
        """
        Summary
        -------
        Draw a contours image obtained using contour(cls, img) class method

        Args:
            img (Any): Source image before contour
            contours (Any): OpenCV image with contours only

        Returns:
            ndarray: Image representation of contours
        """
        dst = np.zeros((img.shape[0], img.shape[1], 1), dtype = "uint8")
        cv.drawContours(dst, contours, -1, (255,0,0), 1)
        return dst

    def align_images(image, template, maxFeatures=500, keepPercent=0.2):
        """
        Summary
        -------
        The alignment method where we want to use contours instead of some 
        interesting points to make process accurate
        
        Parameters
        ----------
        image: Our input photo/scan of a form (such as the IRS W-4). 
        The form itself, from an arbitrary viewpoint, should be identical 
        to the template
        image but with form data present.
        template: The template form image.
        maxFeatures: Places an upper bound on the number of candidate keypoint 
        regions to consider.
        keepPercent: Designates the percentage of keypoint matches to keep, 
        effectively allowing us to eliminate noisy keypoint matching results
        for debugging purposes.
        """
        # convert both the input image and template to grayscale
        imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        templateGray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        contourImage, hierarchyImage = MyAlignment.contour(image)
        contourTemplate, hierarchyTemplate = MyAlignment.contour(template)
        dstImage = MyAlignment.drawContours(image, contourImage)
        dstTemplate = MyAlignment.drawContours(template, contourTemplate)
        # use ORB to detect keypoints and extract (binary) local
        # invariant features
        orb = cv.ORB_create(maxFeatures)
        (kpsA, descsA) = orb.detectAndCompute(dstImage, None)
        (kpsB, descsB) = orb.detectAndCompute(dstTemplate, None)
        # match the features
        method = cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv.DescriptorMatcher_create(method)
        matches = matcher.match(descsA, descsB, None)
        # sort the matches by their distance (the smaller the distance,
        # the "more similar" the features are)
        matches = sorted(matches, key=lambda x:x.distance)
        # keep only the top matches
        keep = int(len(matches) * keepPercent)
        matches = matches[:keep]
        # allocate memory for the keypoints (x, y)-coordinates from the
        # top matches -- we'll use these coordinates to compute our
        # homography matrix
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt
        # compute the homography matrix between the two sets of matched
        # points
        (H, mask) = cv.findHomography(ptsA, ptsB, method=cv.RANSAC)
        # use the homography matrix to align the images
        (h, w) = template.shape[:2]
        aligned = cv.warpPerspective(image, H, (w, h))
        # return the aligned image
        return aligned
