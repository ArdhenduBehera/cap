# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 21:39:13 2018

@author: Andrew
"""
import numpy as np

#get regions of interest of an image (return all possible bounding boxes when splitting the image into a grid)
def getROIS(resolution=33,gridSize=3, minSize=1):
	
	coordsList = []
	step = resolution / gridSize # width/height of one grid square
	
	#go through all combinations of coordinates
	for column1 in range(0, gridSize + 1):
		for column2 in range(0, gridSize + 1):
			for row1 in range(0, gridSize + 1):
				for row2 in range(0, gridSize + 1):
					
					#get coordinates using grid layout
					x0 = int(column1 * step)
					x1 = int(column2 * step)
					y0 = int(row1 * step)
					y1 = int(row2 * step)
					
					if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)): #ensure ROI is valid size
						
						if not (x0==y0==0 and x1==y1==resolution): #ignore full image
							
							#calculate height and width of bounding box
							w = x1 - x0
							h = y1 - y0
							
							coordsList.append([x0, y0, w, h]) #add bounding box to list
	coordsList.append([0, 0, resolution-1, resolution-1])#whole image
	coordsArray = np.array(coordsList)	 #format coordinates as numpy array						

	return coordsArray					
