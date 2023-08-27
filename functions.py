from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
import splitfolders
import glob
import os

def process_image(image):
    ret,bi_inv = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV) #ret is the threshold (127), any above 127 is set to 255
    return bi_inv

""" Otsu """
def process_image_o(image):
    ret, bi_inv = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
    return bi_inv

def process_image_r(image, angle):
    ret, bi_inv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    rows, cols = bi_inv.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)  # Get rotation matrix
    bi_inv = cv2.warpAffine(bi_inv, M, (cols, rows))  # Apply rotation
    return bi_inv

def create_data(tdir_, type_, t_data, pre, rotate_half=False):
    bi_inv_data = []  # Binary inversed - cracks in white
    print('Working On ' + t_data + ' Data: ' + type_ + '\n')
    dir_ = tdir_ + type_ + '/*'
    files = glob.glob(dir_ + '.jpg')
    
    if rotate_half:
        files = np.random.permutation(files)  # Shuffle the file list
    
    for file in files:
        image = cv2.imread(file, 0)  # 0 reads image in grayscale 0-255
        
        if pre == 'otsu':
            bi_inv = process_image_o(image)  # returns binary inversed image-white cracks +  original image
        elif pre == 'rot':
            if rotate_half:
                bi_inv = process_image_r(image, angle=45)  # Rotate the image by 45 degrees
            else:
                bi_inv = process_image_r(image, angle=0)   # No rotation
        else:
            bi_inv = process_image(image)  # returns binary inversed image-white cracks +  original image
        
        bi_inv_data.append(bi_inv)
    
    print('Number of Images Processed: ' + str(len(bi_inv_data)) + '\n')    
    return bi_inv_data


def plot_training_curves(history_dict):
    # Get training/validation accuracy and loss values from history dictionary
    training_accuracy = history_dict['accuracy']
    validation_accuracy = history_dict['val_accuracy']
    training_loss = history_dict['loss']
    validation_loss = history_dict['val_loss']

    # Plot accuracy curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss curves
    plt.subplot(1, 2, 2)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    
def process_image2(image):
    ret,bi_inv = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV) #ret is the threshold (127), any above 127 is set to 255
    return bi_inv, image

""" Takes input: Directory, type pos/neg, range from, range to, Type of data train/test/valid - not important for processing """
def create_data2(dir_):
    colored_data=[]
    bi_inv_data=[] # Binary inversed - cracks in white
    image = cv2.imread(dir_, 0) # 0 reads image in grayscale 0-255
    bi_inv, colored_img = process_image2(image) # returns binary inversed image-white cracks +  original image
    colored_data.append(colored_img)
    bi_inv_data.append(bi_inv)   

    return colored_data, bi_inv_data

def create_data3(folder_path, img_file):
    colored_data=[]
    bi_inv_data=[] # Binary inversed - cracks in white
    dir_ =folder_path+'/'+img_file
    image = cv2.imread(dir_, 0) # 0 reads image in grayscale 0-255
    bi_inv, colored_img = process_image2(image) # returns binary inversed image-white cracks +  original image
    colored_data.append(colored_img)
    bi_inv_data.append(bi_inv)   

    return colored_data, bi_inv_data

""" predicted probability that this image [0] belongs to the positive class """

def predict_image_util(final_pred_inv, model):
    img_test = (final_pred_inv[0].reshape((1, 227, 227, 1)))  
    raw_predicted_label = model.predict(img_test, batch_size=None, verbose=0, steps=None)[0][0]
    
    predicted_label=1;    
    if(raw_predicted_label<0.8):
        predicted_label=0
        
    predicted_label_str='Crack'    
    if(predicted_label==0):
        predicted_label_str='No Crack'
        
    print('Raw Predicted Label(Numeric): '+str(raw_predicted_label))
    print('\nPredicted Label : '+predicted_label_str)    

def predict_image2(dire,img, model):
    dir_ = dire+'/'+img+'.jpg'
    pred_data_colr_, pred_data_inv_ = create_data2(dir_)
    plt.imshow(pred_data_colr_[0])
    pred_data_colr =[]
    pred_data_inv = []
    
    pred_data_inv.append(pred_data_inv_[0])
    pred_data_colr.append(pred_data_colr_[0])
    
    final_pred_colr = np.array(pred_data_colr).reshape(((len(pred_data_colr), 227, 227, 1)))  
    final_pred_inv = np.array(pred_data_inv).reshape(((len(pred_data_inv), 227, 227, 1)))
    predict_image_util(final_pred_inv, model)

def predict_folder(folder_path, model):
    image_names = []
    raw_predictions = []
    image_classifications = []

    for img_file in os.listdir(folder_path):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(folder_path, img_file)
            image_names.append(img_file)

            # Assuming `predict_image_util()` takes an image as input and returns raw_pred and img_classification
            pred_data_colr_, pred_data_inv_ = create_data3(folder_path,img_file)
            pred_data_inv = []
            pred_data_inv.append(pred_data_inv_[0])
            final_pred_inv = np.array(pred_data_inv).reshape(((len(pred_data_inv), 227, 227, 1)))
            raw_pred, img_classification = predict_image_util(final_pred_inv, model)
            raw_predictions.append(raw_pred)
            image_classifications.append(img_classification)

    # Create a pandas DataFrame with the collected data
    data = {
        'Image Name': image_names,
        'Raw Label Prediction': raw_predictions,
        'Image Classification': image_classifications
    }

    df = pd.DataFrame(data)
    return df