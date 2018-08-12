import os
from datetime import datetime
import string
import random

import numpy as np
import keras


def id_generator(size=6, chars=string.ascii_uppercase):
	return ''.join(random.choice(chars) for _ in range(size))


def train_model(model, dataset_fold, learning_rate, batch_size=32, max_no_epoch=100, early_stop_patience=3):
	save_prefix = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '_' + id_generator()
	save_prefix = os.path.join('log', save_prefix)

	model.compile(optimizer = keras.optimizers.Adam(lr=learning_rate),
		loss = 'categorical_crossentropy',
		metrics = ['accuracy'])

	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=0)
	model_cp = keras.callbacks.ModelCheckpoint(save_prefix + '.cp', monitor='val_loss', verbose=0, save_best_only=True)

	results = model.evaluate(dataset_fold['val_data'], dataset_fold['val_labels'], batch_size=batch_size, verbose=0)
	initial_loss = results[0]

	model.save(save_prefix + 'init.cp')

	history = model.fit(x=dataset_fold['train_data'], y = dataset_fold['train_labels'], batch_size = batch_size,
						epochs=max_no_epoch, callbacks = [early_stop, model_cp], verbose=0,
						validation_data = (dataset_fold['val_data'], dataset_fold['val_labels']))

	if min(history.history['val_loss']) > initial_loss:
		no_useful_epochs = 0
		model = keras.models.load_model(save_prefix + 'init.cp')
	else:
		no_useful_epochs = len(history.history['val_loss']) - early_stop_patience - 1
		model = keras.models.load_model(save_prefix + '.cp')

	os.remove(save_prefix + 'init.cp')
	os.remove(save_prefix + '.cp')

	return model


def test_model(model, dataset_fold, batch_size=32):

	preds = model.predict(dataset_fold['test_data'], batch_size=batch_size)
	preds = np.round(preds)

	true_preds = preds[:, 0] == dataset_fold['test_labels'][:, 0]
	acc = float(np.sum(true_preds)) / len(true_preds)

	false_images = dataset_fold['test_image_names'][np.invert(true_preds)]

	return acc, false_images