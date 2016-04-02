# USAGE
# python main_level3.py --source images/ocean_sunset.jpg --target images/ocean_day.jpg

from os.path import basename
from color_transfer import color_transfer
import numpy as np
import argparse
import cv2
import random as rnd 

# --------------------------------------------------------------------
# GLOBALs
source= -1
source_clone= -1
target= -1
target_clone= -1
refPt = []
refPts = []
refPtT = []
refPtsT = []
cropping = False
# --------------------------------------------------------------------

def show_image(title, image, width = 800):
	# resize the image to have a constant width, just to
	# make displaying the images take up less screen real
	# estate
	r = width / float(image.shape[1])
	dim = (width, int(image.shape[0] * r))
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	# show the resized image
	cv2.imshow(title, resized)

# sorted point list -> dictionary of [0,255] keys and [0,100000] values representing cdf
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

# calc initial centroids ---------------------
def find_clstr_stat( boundry, file ):
	num_of_points= ( boundry[1][0] - boundry[0][0] + 1 ) * ( boundry[1][1] - boundry[0][1] + 1 )
	file_channels = cv2.split(file)
	clstr_mu=[0.0,0.0,0.0]
	for ch in range(3):
		ch= file_channels[ch]
		for y in range( boundry[0][1] , boundry[1][1]+1 ):
			for x in range( boundry[0][0] , boundry[1][0]+1 ):
				clstr_mu[ch]+= ch[y][x]
		clstr_mu[ch] /= float(num_of_points)
	clstr_stddev= 0
	clstr_stat= (clstr_mu, clstr_stddev)
	return clstr_stat

def is_dist_close( p, boundry ):
	return True

def is_color_close( pcolor, clstr_stat ):
	return True

def get_cluster( boundry, clstr_stat, file ):
	file_channels = cv2.split(file)
	clstr= []
	for y in file.shape[0]:
		for x in file.shape[1]:
			if y>=boundry[0][1] and y<=boundry[1][1] and x>=boundry[0][0] and x<=boundry[1][0] \
			or  is_color_close( file[y][x], clstr_stat ) and is_dist_close((x,y), boundry):
				clstr+= [(x,y,file[y][x])]
	return clstr

def color_transfer_level3():
	global source_channels, target_channels
	source_copy = source.copy()
	result= []
	for k in range(len(refPts)):	# For each cluster
		pS= [ ( min(refPts[k][0][0], refPts[k][1][0]) ,  min(refPts[k][0][1],refPts[k][1][1]) ),   ( max(refPts[k][0][0],refPts[k][1][0]) , max(refPts[k][0][1],refPts[k][1][1]) ) ]
		pT= [ ( min(refPtsT[k][0][0],refPtsT[k][1][0]) , min(refPtsT[k][0][1],refPtsT[k][1][1]) ), ( max(refPtsT[k][0][0],refPtsT[k][1][0]) , max(refPtsT[k][0][1],refPtsT[k][1][1]) ) ]
		clstr_stat_S= find_clstr_stat( pS, source )
		clstr_stat_T= find_clstr_stat( pS, target )
		cls_k_S= get_cluster( pS, clstr_stat_S, source )
		cls_k_T= get_cluster( pT, clstr_stat_T, target )
		
		for i in range(3):
			"""
			l_s= source_channels[i]
			l_t= target_channels[i]
			len_s= len(l_s)		# e.g. 700
			len_t= len(l_t)		# e.g. 680
			"""
			l_t_flat_cls = map(lambda p:p[2][i], cls_k_T)	
			len_t_flat_cls= len(l_t_flat_cls)	# e.g. 680000
			l_t_flat_cls_sorted = np.sort(l_t_flat_cls)
			hist_dict_t_cls= make_histogram( l_t_flat_cls_sorted )
			hist_dict_t_cls_swap= {v: k for k, v in hist_dict_t_cls.items()}		
	
			l_s_flat_cls = map(lambda p:p[2][i], cls_k_S)
			len_s_flat_cls= len(l_s_flat_cls)
			l_s_flat_cls_sorted = np.sort(l_s_flat_cls)
			hist_dict_s_cls = make_histogram( l_s_flat_cls_sorted )

			for j in range(len(cls_k_S)):
				x= cls_k_S[j][0]
				y= cls_k_S[j][1]
				area_under= hist_dict_s_cls[ cls_k_S[j][2][i] ]  # 202 -> %90.35 -> 90350/100000 -> 90350
				matching_area= min(hist_dict_t_cls.values(), key=lambda x:abs(x-area_under))  # 90352 is found
				source_copy[y][x][i]= hist_dict_t_cls_swap[matching_area]	# 215 

	transfer = source_copy
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	return transfer

#-------------------------------------





# --------------------------------------------------------------------

source= -1
source_clone= -1
target= -1
target_clone= -1
refPt = []
refPts = []
refPtT = []
refPtsT = []
cropping = False

def click_to_select_source_boundries(event, x, y, flags, param):
	(r,g,b)= source[y,x]
	global refPt, refPts, cropping, source_clone
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		refPts+= [refPt]
		print refPts
		cropping = False
		cv2.rectangle(source_clone, refPt[0], refPt[1], (255-r,255-g,255-b), 2)
		cv2.imshow("source_clone", source_clone)
		refPt = []

def click_to_select_target_boundries(event, x, y, flags, param):
	(r,g,b)= source[y,x]
	global refPtT, refPtsT, cropping, target_clone
	if event == cv2.EVENT_LBUTTONDOWN:
		refPtT = [(x, y)]
		cropping = True
	elif event == cv2.EVENT_LBUTTONUP:
		refPtT.append((x, y))
		refPtsT+= [refPt]
		print refPtsT
		cropping = False
		cv2.rectangle(source_clone, refPtT[0], refPtT[1], (255-r,255-g,255-b), 2)
		cv2.imshow("target_clone", target_clone)
		refPtT = []

def select_source_clusters_boundries():
	print "select_source_clusters_boundries()"
	global source_clone
	source_clone = source.copy()
	cv2.namedWindow("source_clone")
	cv2.setMouseCallback("source_clone", click_to_select_source_boundries)
	while True:					# keep looping until the 'q' key is pressed
		print "select_source_clusters_boundries() WHILE"
		cv2.imshow("source_clone", source_clone)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("r"):		# if the 'r' key is pressed, reset the cropping region
			source_clone = source.copy()
		elif key == ord("c"):	# if the 'c' key is pressed, break from the loop
			break

def select_target_clusters_boundries():
	print "select_source_clusters_boundries()"
	global target_clone
	target_clone = target.copy()
	cv2.namedWindow("target_clone")
	cv2.setMouseCallback("target_clone", click_to_select_target_boundries)
	while True:					# keep looping until the 'q' key is pressed
		print "select_target_clusters_boundries() WHILE"
		cv2.imshow("target_clone", source_clone)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("r"):		# if the 'r' key is pressed, reset the cropping region
			target_clone = target.copy()
		elif key == ord("c"):	# if the 'c' key is pressed, break from the loop
			break

def run_app():
	print "run_app()"
	global source, target
	ap = argparse.ArgumentParser() 	# construct the argument parser and parse the arguments
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
	
	print "source (H,W,Ch): ", source.shape
	print "target (H,W,Ch): ", target.shape

	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
	print "source first pixel LAB: ", source[1,1] 

	# (l_s, a_s, b_s) = cv2.split(source)
	# (l_t, a_t, b_t) = cv2.split(target)
	source_channels = cv2.split(source)
	target_channels = cv2.split(target)
	
	select_source_clusters_boundries()	# refPts is ready
	select_target_clusters_boundries()	# refPtsT is ready
	
	"""
	if len(refPt) == 2:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		cv2.imshow("ROI", roi)
		cv2.waitKey(0)
	"""
	cv2.destroyAllWindows()	# close all open windows
	
	# transfer the color distribution from the source image to the target image
	transfer = color_transfer_level3()
	"""
	show_image("Source", source)
	show_image("Target", target)
	show_image("Transfer", transfer)
	"""
	# write to file
	# cv2.imwrite(outname, transfer)
	cv2.waitKey(0)

# -------------RUN APP------------------------------------------
run_app()






