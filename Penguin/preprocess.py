import cv2
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Penguin:
    """
    # Rename_data function:  @Done  **notice that this function needs to be maniplated under different folder**
        requires 2 arguments [path,preName] , path will be the 
        data folder path ,and preName will be the first name
        you want to rename the data. It will rename all data
        which in the folder with preName_0 1 2 3 ... etc.

    # Load_images function: @Done
        requires 1 argument [path], path will be the path of dataset
        folder, and it returns one list containing two lists:
        [[image_list],[image_list]]
    
    # Str_to_categorical function: @Done

    # Resize function: @Done
        requires 3 arguments [width,height,image_list] and the 
        variable type of image_list has to be list. This function
        returns resized_image_list which has dataType list too. 

    # Gaussian_blur: @Done

    # Display_one function: @Done

    # Display_two function: @Done
    """
    def rename_data(self,path,pre_name):
        ii = 0
        image_files = [i for i in listdir(path)] #load all images which are on the same path
        for i in image_files:
            file_path = path + r'\ ' + i
            file_path = file_path.replace(" ", "")
            os.rename(file_path,path+'\{}_{}.jpg'.format(pre_name,ii))
            ii += 1
    
    def load_images(self,path):
        image_list = []
        image_target = []
        image_files = [i for i in listdir(path)]
        for i in image_files:
            image_path = path + '\ ' + i
            image_path = image_path.replace(' ','')
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_list.append(image)
            image_target.append(i[:i.index('_')])
        all_in_one = [image_list,image_target]
        return all_in_one

    def str_to_categorical(self,input_target_list):
        max = 0
        unique_list = []
        index_list = []
        for i in input_target_list:
            if i not in unique_list:
                unique_list.append(i)
        for i in input_target_list:
            if max < unique_list.index(i):
                max = unique_list.index(i)
            index_list.append(unique_list.index(i))
        matrix = np.zeros(len(index_list)*(max+1)).reshape(len(index_list),max+1)
        for i in range(0,len(index_list)):
            matrix[i,index_list[i]] = 1
        return matrix
    
    def resize(self,input_image_list,width,height):
        dim = (width, height)
        output_image_list = []
        for i in input_image_list:
            res = cv2.resize(i, dim, interpolation=cv2.INTER_LINEAR)
            output_image_list.append(res)
        return output_image_list
    
    def Gaussian_blur(self,input_image_list,kernel_size = 5):
        output_image_list = []
        for i in input_image_list:
            blur = cv2.GaussianBlur(i, (kernel_size,kernel_size), 0)
            output_image_list.append(blur)
        return output_image_list

    def Segmentation(self,input_image_list):
        output_image_list = []
        for i in input_image_list:
            gray = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            output_image_list.append(thresh)
        return output_image_list

    def display_one(self,image_list,index,title = 'Picture'):
        plt.imshow(image_list[index]), plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    def display_two(self,original_image_list,edited_image_list,index,title1 = "Original", title2 = "Edited"):
        plt.subplot(121), plt.imshow(original_image_list[index]), plt.title(title1)
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edited_image_list[index]), plt.title(title2)
        plt.xticks([]), plt.yticks([])
        plt.show()
    
    #def load_image(self,):



