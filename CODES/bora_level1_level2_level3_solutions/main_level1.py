# Bora
from os.path import basename

# USAGE
# python bb_main.py --source images/ocean_sunset.jpg --target images/ocean_day.jpg

# import the necessary packages
from color_transfer import color_transfer
import numpy as np
import argparse
import cv2
#-------------------------------------

def color_transfer_level1(source, target):
	print "source img width: ", source.shape[1]
	print "source img height: ", source.shape[0]
	print "source img channels: ", source.shape[2]
	print "source first pixel RGB: ", source[1,1]
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
	print "source first pixel LAB: ", source[1,1] 

	"""
	# Backward transformation
	source = cv2.cvtColor(source.astype("uint8"), cv2.COLOR_LAB2BGR)
	target = cv2.cvtColor(target.astype("uint8"), cv2.COLOR_LAB2BGR)
	print "source first pixel RGB: ", source[1,1]
	"""
	# compute color statistics for the source and target images
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

	# subtract the means from the target image
	(l, a, b) = cv2.split(source)
	l -= lMeanSrc
	a -= aMeanSrc
	b -= bMeanSrc

	# scale by the standard deviations
	l = (lStdTar / lStdSrc) * l
	a = (aStdTar / aStdSrc) * a
	b = (bStdTar / bStdSrc) * b

	# add in the source mean
	l += lMeanTar
	a += aMeanTar
	b += bMeanTar

	# clip the pixel intensities to [0, 255] if they fall outside
	# this range
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)

	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	# return the color transferred image
	return transfer

def image_stats(image):
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	print "l:\n", l 
	print "a:\n", a 
	print "b:\n", b 
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())
	print "(lMean, lStd): ", (lMean, lStd)
	print "(aMean, aStd): ", (aMean, aStd)
	print "(bMean, bStd): ", (bMean, bStd)

	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)


#-------------------------------------

def show_image(title, image, width = 800):
	# resize the image to have a constant width, just to
	# make displaying the images take up less screen real
	# estate
	r = width / float(image.shape[1])
	dim = (width, int(image.shape[0] * r))
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

	# show the resized image
	cv2.imshow(title, resized)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required = True,	help = "Path to the source image")
ap.add_argument("-t", "--target", required = True,	help = "Path to the target image")
ap.add_argument("-o", "--output", help = "Path to the output image (optional)")
args = vars(ap.parse_args())

# load the images
source = cv2.imread(args["source"])
target = cv2.imread(args["target"])
outname= "level1_"+basename(args["source"]).split(".")[0]+"_to_"+basename(args["target"]).split(".")[0]+".jpg"
if args["output"] is not None:
	outname= args["output"]

# transfer the color distribution from the source image to the target image
transfer = color_transfer_level1(source, target)

"""
# show the images and wait for a key press
show_image("Source", source)
show_image("Target", target)
show_image("Transfer", transfer)
"""

# write to file
cv2.imwrite(outname, transfer)

cv2.waitKey(0)











