import os
from random import shuffle

import numpy as np
from scipy.io import loadmat
from sklearn.svm import LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import zscore


def generate_fold_indices(no_examples_per_class=np.asarray([2000, 4000]), no_class=2, no_folds=10,
					train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
		"""
		Generates indices for k-fold cross validation.
		"""
		assert(test_ratio * no_folds == 1)
		no_train = np.round(no_examples_per_class * train_ratio).astype(np.int)
		no_val = np.round(no_examples_per_class * val_ratio).astype(np.int)
		no_test = (no_examples_per_class - no_train - no_val).astype(np.int)

		indices = {'train': np.empty((no_folds, sum(no_train)), dtype=np.int),
					'val': np.empty((no_folds, sum(no_val)), dtype=np.int),
					'test': np.empty((no_folds, sum(no_test)), dtype=np.int)}

		

		shuffled_inds = [range(no_examples_per_class[0]), no_examples_per_class[0] + range(no_examples_per_class[1])]
		shuffle(shuffled_inds[0])
		shuffle(shuffled_inds[1])

		for ind_fold in range(no_folds):
			# class #1
			copied_shuffled_inds = list(shuffled_inds[0])

			indices['test'][ind_fold, :no_test[0]] = copied_shuffled_inds[ind_fold * no_test[0]:(ind_fold + 1) * no_test[0]]
			del copied_shuffled_inds[ind_fold * no_test[0]:(ind_fold + 1) * no_test[0]]

			shuffle(copied_shuffled_inds)
			indices['train'][ind_fold, :no_train[0]] = copied_shuffled_inds[:no_train[0]]
			indices['val'][ind_fold, :no_val[0]] = copied_shuffled_inds[no_train[0]:]

			# class #2
			copied_shuffled_inds = list(shuffled_inds[1])

			indices['test'][ind_fold, no_test[0]:] = copied_shuffled_inds[ind_fold * no_test[1]:(ind_fold + 1) * no_test[1]]
			del copied_shuffled_inds[ind_fold * no_test[1]:(ind_fold + 1) * no_test[1]]

			shuffle(copied_shuffled_inds)
			indices['train'][ind_fold, no_train[0]:] = copied_shuffled_inds[:no_train[1]]
			indices['val'][ind_fold, no_val[0]:] = copied_shuffled_inds[no_train[1]:]

		return indices


def main():
	domains = ['IR', 'VL']
	preprocess_methods = ['mean_subtraction', 'scaling']
	#init_methods = ['random', 'ImageNet']
	init_methods = ['ImageNet']
	model_names = ['ResNet50', 'VGG19']
	stages = ['stage_1', 'stage_2', 'stage_3', 'stage_4', 'stage_5']
	#sampling_method = ['pca', 'uniform']
	sampling_method = ['uniform']
	classifiers = ['SVM', 'NB', 'RF']

	indices = generate_fold_indices()
	f = open(os.path.join(log_path, 'log.txt'), 'w')

	for sampling_method in sampling_method:
		out_dict = loadmat('feats_' + sampling_method + '.mat')
		print sampling_method
		f.write(sampling_method + '\n')
		for domain in domains:
			for preprocess_method in preprocess_methods:
				for init_method in init_methods:
					for model_name in model_names:
						for ind_stage, stage in enumerate(stages):
							name = domain + '_' + preprocess_method + '_' + init_method + '_' + model_name + '_' + stage
							print name
							f.write(name + '\n')
							feats = out_dict[name + '_feats']
							labels = out_dict[name + '_labels']

							for classifier in classifiers:
								conf_mat = [0, 0, 0, 0] # TN, FP, FN, TP
								for ind_fold in range(10):
									inds_train = np.concatenate((indices['train'][ind_fold], indices['val'][ind_fold]))
									inds_test = indices['test'][ind_fold]

									feats_train = feats[inds_train]
									labels_train = labels[inds_train, 0]
									feats_test = feats[inds_test]
									labels_test = labels[inds_test, 0]
									
									if classifier == 'SVM':
										clf = LinearSVC()
									elif classifier == 'NB':
										clf = GaussianNB()
									elif classifier == 'RF':
										clf = RandomForestClassifier()

									clf.fit(feats_train, labels_train)
									preds = clf.predict(feats_test)

									asd = feats_train[0,:]
									asd = asd[np.newaxis, :]
									import time
									start = time.time()
									clf.predict(asd)
									end = time.time()
									print(end - start)
									print classifier

									preds = preds.astype(np.bool)
									labels_test = labels_test.astype(np.bool)
									conf_mat[0] += np.sum(np.invert(preds) & np.invert(labels_test))
									conf_mat[1] += np.sum(preds & np.invert(labels_test))
									conf_mat[2] += np.sum(np.invert(preds) & labels_test)
									conf_mat[3] += np.sum(preds & labels_test)

								percent_error = 100 - ((conf_mat[0] + conf_mat[3]) / float(sum(conf_mat))) * 100
								print classifier + ' ' + str(percent_error)
								#print conf_mat
								f.write(classifier + ' ' + str(percent_error) + '\n')
								#f.write(str(conf_mat) + '\n')
	f.close()


if __name__ == "__main__":
	log_path = os.path.join('log', 'feat_class')
	if not os.path.isdir(log_path):
		os.mkdir(log_path)
	main()