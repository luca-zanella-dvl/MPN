from utils import *
import os
import cv2
import pickle
import numpy as np

if __name__=="__main__":

    with open('err_dict.pkl', 'rb') as file:
    
        err_dict = pickle.load(file)
    
    endmin = [1]*3
    endmax = [-1]*3
    for key, value in err_dict.items():
        value = value*100
        
        mmin = [ np.min(value[i]) for i in range(3)]
        mmax = [ np.max(value[i]) for i in range(3)]
        
        for k in range(3):
            if mmin[k]< endmin[k]: 
                endmin[k] = mmin[k]
            if mmax[k]> endmax[k]: 
                endmax[k] = mmax[k]
        
    
    

    for key, value in err_dict.items():
        for k in range(3):
            
            value[k] = ((value[k] - endmin[k])/ (endmax[k] - endmin[k]))

        
        value = np.mean((value*255), axis = 0).astype(np.uint8)        
        visualize_pred_err(key , value, 256, 256 , output_dir="heatmap")
       

