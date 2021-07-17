import unittest
from os import path
import cv2 as cv
import os
import numpy as np
from src.MyAlignmentMethod import MyAlignment
from pkg_resources import resource_filename

class TestMyAlignmentMethod(unittest.TestCase):

    def test_loadImage(self):
        myCv = MyAlignment
        imgPath = resource_filename('tests.resources', 'LENA.PNG')
        img = myCv.loadImage(imgPath)
        self.assertFalse(img is None)
        imgPath = resource_filename('tests.resources', 'LENA.tif')
        img = myCv.loadImage(imgPath)
        self.assertFalse(img is None)
        imgPath = resource_filename('tests.resources', 'M57_RAW.fit')
        img = myCv.loadImage(imgPath)
        self.assertTrue(img is None)

    def test_loadFitImage(self):
        myCv = MyAlignment
        imgPath = resource_filename('tests.resources', 'M57_RAW.fit')
        img = myCv.loadFitImage(imgPath)
        self.assertFalse(img is None)

    def test_FitAsOpenCvImage(self):
        myCv = MyAlignment
        imgPath = resource_filename('tests.resources', 'M57_RAW.fit')
        fit = myCv.loadFitImage(imgPath)
        img = myCv.fitAsOpenCVImage(fit)
        self.assertFalse(img is None)

    def test_loadImageOverloaded(self):
        myCv = MyAlignment
        imgPath = resource_filename('tests.resources', 'LENA.PNG')
        img = myCv.loadImage(imgPath)
        self.assertFalse(img is None)
        img = myCv.loadImage(imgPath, False)
        self.assertFalse(img is None)
        imgPath = resource_filename('tests.resources', 'M57_RAW.fit')
        img = myCv.loadImage(imgPath, True)
        self.assertFalse(img is None)

    def test_saveImage(self):
        myCv = MyAlignment
        imgPath = resource_filename('tests.resources', 'M57_RAW.fit')
        fit = myCv.loadFitImage(imgPath)
        img = myCv.fitAsOpenCVImage(fit)
        absolutePath = 'C:/Users/3D/Desktop/M57.png'
        myCv.saveImage(img, absolutePath)
        self.assertTrue(path.exists(absolutePath))
        os.remove(absolutePath)

    def test_sobel(self):
        myCv = MyAlignment
        imgPath = resource_filename('tests.resources', 'M57_RAW.fit')
        fit = myCv.loadFitImage(imgPath)
        img = myCv.fitAsOpenCVImage(fit)
        sobel = myCv.sobel(img, cv.CV_16S, 1, 0)
        self.assertFalse((sobel is None))
        imgPath = resource_filename('tests.resources', 'LENA.PNG')
        img = myCv.loadImage(imgPath)
        sobel = myCv.sobel(img, cv.CV_16S, 1, 0)
        self.assertFalse((sobel is None))

    def test_contour(self):
        myCv = MyAlignment
        imgPath = resource_filename('tests.resources', 'M57_RAW.fit')
        fit = myCv.loadFitImage(imgPath)
        img = myCv.fitAsOpenCVImage(fit)
        image = img
        contours,hierarchy = myCv.contour(img)
        self.assertFalse((contours is None))
        self.assertFalse((hierarchy is None))
        imgPath = resource_filename('tests.resources', 'LENA.PNG')
        img = myCv.loadImage(imgPath)
        image = img
        contours,hierarchy = myCv.contour(img)
        self.assertFalse((contours is None))
        self.assertFalse((hierarchy is None))

    def test_drawContour(self):
        myCv = MyAlignment
        imgPath = resource_filename('tests.resources', 'LENA.PNG')
        img = myCv.loadImage(imgPath)
        image = myCv.drawContours(img, None)
        self.assertFalse(image is None)
        imgPath = resource_filename('tests.resources', 'M57_RAW.fit')
        fit = myCv.loadFitImage(imgPath)
        img = myCv.fitAsOpenCVImage(fit)
        ctrs,hierarchy = myCv.contour(img)
        image = myCv.drawContours(img, ctrs)
        self.assertFalse(image is None)

    def test_alignImage(self):
        myCv = MyAlignment
        templatePath = resource_filename('tests.resources', 'M57_RAW.fit')
        imgPath = resource_filename('tests.resources', 'M57_00001.fit')
        template = myCv.loadImage(templatePath, True)
        img = myCv.loadImage(imgPath, True)
        aligned = myCv.align_images(img, template)
        self.assertFalse(aligned is None)
        stacked = np.hstack([aligned, template])
        self.assertFalse(stacked is None)

if __name__ == '__main__':
    unittest.main()