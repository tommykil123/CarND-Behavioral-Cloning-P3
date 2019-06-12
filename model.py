import numpy as np
import csv
import cv2
import sklearn
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from NVIDIA_CNN import model
from random import shuffle
from math import ceil

class Pipeline:
    def __init__(self, model=None, base_path=''):
        self.data = []#
        self.model = model
        self.model_name = 'current_model'
        self.epochs = 10
        self.SAS_corr = 0.08
        self.batch_size = 128
        self.base_path = base_path
        self.training_samples = []
        self.validation_samples = []
        self.image_path = self.base_path + '/IMG/'
        self.csv_path = self.base_path + '/driving_log.csv'
        
    def getSamples(self):
        print('Running getSamples functions')
        with open(self.csv_path) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                self.data.append(line)
        return None
    
    def split_data(self):
        train_samples, validation_samples = train_test_split(self.data, test_size=0.2)
        self.training_samples = train_samples
        self.validation_samples = validation_samples
        return None
    
    def generator(self, samples, batch_size=32):
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    for i in range(3):
                        name = self.image_path + batch_sample[i].split('/')[-1]
                        image = cv2.imread(name)
                        image_flip = cv2.flip(image,1)

                        if (i == 0):
                            angle = float(batch_sample[3])
                        elif (i == 1):
                            angle = float(batch_sample[3]) + self.SAS_corr
                        elif (i == 2):
                            angle = float(batch_sample[3]) - self.SAS_corr
                        angle_flip = -1.0 * angle

                        images.append(image)
                        images.append(image_flip)
                        angles.append(angle)
                        angles.append(angle_flip)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)
                
    def train_generator(self, batch_size):
        return self.generator(samples = self.training_samples, batch_size = self.batch_size)
    def validation_generator(self, batch_size):
        return self.generator(samples = self.validation_samples, batch_size = self.batch_size)
    
    def run(self):
        self.split_data()
        checkpoint = ModelCheckpoint('models/model_best_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
        callbacks_list = [checkpoint]
        train_len = len(self.training_samples)
        print(train_len)
        valid_len = len(self.validation_samples)
        print(valid_len)
        history = self.model.fit_generator(self.train_generator(self.batch_size),
            steps_per_epoch=ceil(train_len/self.batch_size),
            validation_data=self.validation_generator(self.batch_size),
            validation_steps=ceil(valid_len/self.batch_size),
            callbacks=callbacks_list,
            epochs=self.epochs, verbose=1)
        self.model.save('models/' + self.model_name +'.h5')
        return (history)
    
def plotError(history):
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show();  
    plt.savefig('Training vs Test Loss.png')
    
def main():
    print('initializing pipeline instance')
    pipeline = Pipeline(model=model(), base_path='data')
    print('getting samples from the data folder')
    pipeline.getSamples()
    print('running the pipeline...')
    history = pipeline.run()
    plotError(history)
    
if __name__ == '__main__':
    main()
         