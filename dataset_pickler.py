import os
import shutil
import zipfile
import pickle

import numpy as np
from PIL import Image


def pickle_dataset():
	"""
	Downloads the dataset, extracts it, reads the images and their names, pickles them
	The images are not preprocessed, and kept as uint8
	It calculates the mean pixel value for normalization
	"""
	dataset_names = ['Power_Line_Database (Infrared-IR and Visible Light-VL).zip',
					'TY_IR_2000_extra.zip?dl=1',
					'TY_VL_2000_extra.zip?dl=1']

	dataset_urls = ['https://data.mendeley.com/datasets/n6wrv4ry6v/7/files/472605f5-80bc-4dca-8052-d8ca17cc618f/Power_Line_Database%20(Infrared-IR%20and%20Visible%20Light-VL).zip',
					'https://data.mendeley.com/datasets/wzz5pf7h4m/1/files/4a386879-2e2b-47bd-bcc2-146b596730f7/TY_IR_2000_extra.zip?dl=1',
					'https://data.mendeley.com/datasets/wzz5pf7h4m/1/files/488e988f-282c-4b64-8f48-797ddc259b18/TY_VL_2000_extra.zip?dl=1']

	for ind_dataset, dataset_name in enumerate(dataset_names):
		# download the zip file
		os.system("wget -t0 -c '" + dataset_urls[ind_dataset] +  "'")
		# extract the dataset
		if os.path.isdir(dataset_name):
			shutil.rmtree(dataset_name)
		with zipfile.ZipFile(dataset_name, 'r') as z:
			z.extractall()

	# simplify directory names
	if os.path.isdir('PL_raw'):
		shutil.rmtree('PL_raw')
	os.rename('Power_Line_Database (Infrared-IR and Visible Light-VL)', 'PL_raw')
	os.rename(os.path.join('PL_raw', 'Infrared (IR)'), os.path.join('PL_raw', 'IR'))
	os.rename(os.path.join('PL_raw', 'Visible Light (VL)'), os.path.join('PL_raw', 'VL'))

	# move the extra negative examples
	source_paths = ['TY_IR_2000_extra', 'TY_VL_2000_extra']
	target_paths = [os.path.join('PL_raw', 'IR'), os.path.join('PL_raw', 'VL')]

	for ind in range(2):
		file_names = os.listdir(source_paths[ind])
		for file_name in file_names:
			shutil.move(os.path.join(source_paths[ind], file_name), target_paths[ind])
		shutil.rmtree(source_paths[ind])

	# remove trash
	shutil.rmtree('__MACOSX')
	os.remove(os.path.join('PL_raw', '.DS_Store'))
	os.remove(os.path.join('PL_raw', 'Licence.txt'))
	os.remove(os.path.join('PL_raw', 'IR', '.DS_Store'))
	os.remove(os.path.join('PL_raw', 'VL', '.DS_Store'))

	# read the images into numpy arrays
	domains = {'IR', 'VL'}
	classes = {'T', 'F'}
	no_examples_per_class = [2000, 4000]
	image_size = [128, 128, 3]
	
	images = {}
	labels = {}
	image_names = {}
	pixel_mean = {}
	
	for domain in domains:
		images_in_domain = np.ndarray([sum(no_examples_per_class)] + image_size, dtype=np.uint8)
		labels_in_domain = np.ndarray((sum(no_examples_per_class), 2), dtype=np.uint8)
		image_names_in_domain = np.ndarray(sum(no_examples_per_class), dtype=object)
		
		file_names = os.listdir(os.path.join('PL_raw', domain))
		file_names.sort()
		
		for ind_file, file_name in enumerate(file_names):
			full_file_path = os.path.join('PL_raw', domain, file_name)
			image = Image.open(full_file_path)
			np_image = np.fromstring(image.tobytes(), dtype=np.uint8)
			np_image = np_image.reshape((image.size[1], image.size[0], 3))
			images_in_domain[ind_file] = np_image
			if ind_file < no_examples_per_class[0]:
				labels_in_domain[ind_file] = [1, 0]
			else:
				labels_in_domain[ind_file] = [0, 1]
			image_names_in_domain[ind_file] = file_name
		
		
		mean_pixels_of_images = np.mean(images_in_domain, axis=(1, 2))
		pixel_mean_in_domain = np.mean(mean_pixels_of_images, axis=0)
		
		images[domain] = images_in_domain
		labels[domain] = labels_in_domain
		image_names[domain] = image_names_in_domain
		pixel_mean[domain] = pixel_mean_in_domain
		
	dataset_info = 'Power Line Database v7 + extra negative examples\n'\
					'2000 positive and 4000 negative examples for each domain\n'\
					'IR (infrared) and VL (visible light)\n'\
					'label [1, 0] denotes the existence of power lines, and vice versa'

	# pickle the images
	d = {'images': images, 'labels': labels, 'pixel_mean': pixel_mean,
		'image_names': image_names, 'dataset_info': dataset_info}
	with open('power_lines.p', 'wb') as f:
		pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

	# clean up
	shutil.rmtree('PL_raw')


def main():
	if not os.path.isfile('power_lines.p'):
		print 'Pickling the raw dataset'
		pickle_dataset()
	

if __name__ == "__main__":
	main()