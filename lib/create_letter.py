import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# aman gulati

def create_letter(a, add_noise = False):

    if a == 'e':
        aux = [10,11,12,13,18,26,27,28,29,34,42,50,51,52,53]
    if a == 'l':
        aux = [10,18,26,34,42,50,51,52,53]
    if a == 'f':
        aux = [10,11,12,13,18,26,27,28,29,34,42,50,]
        
    letter = np.zeros((8*8))
    letter = letter + 1
    letter[aux] = 0
    
    if add_noise == True:
        letter = np.add(letter, -np.random.normal(scale = 0.2, size = 64))
        #letter[letter < 0] = 0 
    
    return letter


#lss = create_letter('f', add_noise = True)
#print(lss)

#print(lss.shape)
#plt.imshow(lss.reshape(8,8), cmap='gray')
#plt.show()