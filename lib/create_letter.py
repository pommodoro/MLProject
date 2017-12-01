
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

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
        letter[letter < 0] = 0 
    
    return letter

#Getting the data
lists_e=[]
lists_f=[]
lists_l=[]
for i in range(2000):
    lists_e.append(create_letter('e',True))  #For E
    lists_f.append(create_letter('f',True))  #For F
    lists_l.append(create_letter('l',True))  #For L
inp_e=np.array(lists_e)
inp_f=np.array(lists_f)
inp_l=np.array(lists_l)

#Final data array
data=np.concatenate((inp_e,inp_f,inp_l))  

#Labels (1 for E, 2 for F, 3 for L)
labels=np.concatenate((np.tile(1,[2000]),np.tile(2,[2000]),np.tile(3,[2000])))

#lss = create_letter('f', add_noise = True)
#print(lss)

#print(lss.shape)
#plt.imshow(lss.reshape(8,8), cmap='gray')
#plt.show()

