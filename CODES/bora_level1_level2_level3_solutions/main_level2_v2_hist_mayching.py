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

def make_histogram(l_flat_sorted):
	len_flat= len(l_flat_sorted)
	hist_dict = {}
	for i in range(len(l_flat_sorted)):
		val= l_flat_sorted[i]
		if i < len(l_flat_sorted)-1 and val==l_flat_sorted[i+1]: 
			continue
		thousandth= int(100000*float(i)/float(len_flat))
		if not val in hist_dict:
			hist_dict[val]= thousandth # 28 -> 52th in 1000 (0.052), # 250 -> 96500th in 100000 (0.96500)
	print hist_dict
	return hist_dict


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
	# (l_s, a_s, b_s) = cv2.split(source)
	# (l_t, a_t, b_t) = cv2.split(target)
	source_channels = cv2.split(source)
	target_channels = cv2.split(target)

	result= []
	for i in range(3):
		print "i:", i
		l_s= source_channels[i]
		l_t= target_channels[i]
		len_s= len(l_s)
		len_t= len(l_t)
		print "len_s: ", len_s, "\nlen_t: ", len_t 
	
		# print "l_t:", l_t
		l_t_flat = l_t.flatten()
		# print "l_t_flat:", l_t_flat
		len_t_flat= len(l_t_flat)
		print "len_t_flat:", len_t_flat
		l_t_flat_sorted = np.sort(l_t_flat)
		# print "l_t_flat_sorted:", l_t_flat_sorted
		hist_dict_t= make_histogram(l_t_flat_sorted)
		hist_dict_t_swap= {v: k for k, v in hist_dict_t.items()}		

		# print "l_s:", l_s
		l_s_flat = l_s.flatten()
		# print "l_s_flat:", l_s_flat
		len_s_flat= len(l_s_flat)
		print "len_s_flat:", len_s_flat
		l_s_flat_sorted = np.sort(l_s_flat)
		# print "l_s_flat_sorted:", l_s_flat_sorted
		hist_dict_s = make_histogram(l_s_flat_sorted)

		for i in range(len(l_s_flat)):
			area_under= hist_dict_s[ l_s_flat[i] ]  # %90.35 -> 90350/100000 -> 90350
			matching_area= min(hist_dict_t.values(), key=lambda x:abs(x-area_under))  # 90352 is found
			l_s_flat[i] = hist_dict_t_swap[matching_area]
			"""
			for key_val,val_area in hist_dict_t.iteritems():
				if val_area == matching_area:
					 l_s_flat[i] = key_val
			"""
		l_s= np.reshape(l_s_flat, (-1,source.shape[1] ))
		print "ls.shape:", l_s.shape
		
		# clip the pixel intensities to [0, 255] if they fall outside
		# this range
		l_s = np.clip(l_s, 0, 255)
		result+= [l_s]
	print result
	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge(result)
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	# return the color transferred image
	return transfer


def image_stats(image):
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	"""
	print "l:\n", l 
	print "a:\n", a 
	print "b:\n", b 
	"""
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())
	"""
	print "(lMean, lStd): ", (lMean, lStd)
	print "(aMean, aStd): ", (aMean, aStd)
	print "(bMean, bStd): ", (bMean, bStd)
	"""
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
outname= "level2_v2_"+basename(args["source"]).split(".")[0]+"_to_"+basename(args["target"]).split(".")[0]+".jpg"
if args["output"] is not None:
	outname= args["output"]

# transfer the color distribution from the source image to the target image
transfer = color_transfer_level1(source, target)

# show the images and wait for a key press
"""
show_image("Source", source)
show_image("Target", target)
show_image("Transfer", transfer)
"""
# write to file
cv2.imwrite(outname, transfer)

cv2.waitKey(0)











