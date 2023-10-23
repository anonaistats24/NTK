#!/usr/bin/env python
# coding: utf-8

# this script draws scatter plots showing decay rate of eigenvalues

# In[1]:


import numpy as np
from numpy import linalg as LA

dimension = 3

def kernel_eig_fast (activation):
    N = 10000
    M = 1000000
    A = np.zeros((N,N))
    x = np.random.normal(0, 1, size=(N, dimension))
    x = x/np.reshape(np.sqrt(np.sum(np.multiply(x,x), axis = 1)), [-1,1])
    y = np.random.normal(0, 1, size=(dimension, M))
    points_vs_feat = activation(np.matmul(x,y))
    points_vs_feat = np.matmul(points_vs_feat,points_vs_feat.T)/M
    return LA.eigh(points_vs_feat)


# In[ ]:


relu = lambda x: x * (x > 0) #np.maximum(x,x*0.0)
eigenvalues1, _ = kernel_eig_fast(relu)
eigenvalues1 = np.sort(eigenvalues1)
eigenvalues1 = eigenvalues1[::-1]


# In[ ]:


sigma_1 = lambda x: x * (x > 0) - (x-1) * (x > 1) #(np.maximum(x,x*0.0)-np.maximum(x-1.0,x*0.0))
eigenvalues2, _ = kernel_eig_fast(sigma_1)
eigenvalues2 = np.sort(eigenvalues2)
eigenvalues2 = eigenvalues2[::-1]


# In[ ]:


exp = lambda x: np.exp(-0.5*np.multiply(x,x))
eigenvalues, _ = kernel_eig_fast(exp)
eigenvalues = np.sort(eigenvalues)
eigenvalues = eigenvalues[::-1]


# In[ ]:


sigmoid = lambda x: 1 / (1 + np.exp(-x))
eigenvalues3, _ = kernel_eig_fast(sigmoid)
eigenvalues3 = np.sort(eigenvalues3)
eigenvalues3 = eigenvalues3[::-1]


# In[ ]:


tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
eigenvalues4, _ = kernel_eig_fast(tanh)
eigenvalues4 = np.sort(eigenvalues4)
eigenvalues4 = eigenvalues4[::-1]


# In[ ]:


# Import necessary libraries
import seaborn as sns
import pandas as pd 
from sklearn.linear_model import LinearRegression




dataset = pd.DataFrame({'log(rank)': np.log(range(1,1001)), 'log(eig)': np.log(eigenvalues1[:1000])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,1001)), [-1,1]),                              np.reshape(np.log(eigenvalues1[:1000]), [-1,1]))
plt.legend(title='relu\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_relu.png', dpi=400, bbox_inches='tight')
plt.show()


dataset = pd.DataFrame({'log(rank)': np.log(range(1,1001)), 'log(eig)': np.log(eigenvalues2[:1000])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,1001)), [-1,1]),                              np.reshape(np.log(eigenvalues2[:1000]), [-1,1]))
plt.legend(title='σ_1\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_sigma1.png', dpi=400, bbox_inches='tight')
plt.show()

dataset = pd.DataFrame({'log(rank)': np.log(range(1,1001)), 'log(eig)': np.log(eigenvalues[:1000])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,1001)), [-1,1]),                              np.reshape(np.log(eigenvalues[:1000]), [-1,1]))
plt.legend(title='Gaussian\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_gauss.png', dpi=400, bbox_inches='tight')
plt.show()

dataset = pd.DataFrame({'log(rank)': np.log(range(1,1001)), 'log(eig)': np.log(eigenvalues3[:1000])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,1001)), [-1,1]),                              np.reshape(np.log(eigenvalues3[:1000]), [-1,1]))
plt.legend(title='sigmoid\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_sigmoid.png', dpi=400, bbox_inches='tight')
plt.show()

dataset = pd.DataFrame({'log(rank)': np.log(range(1,1001)), 'log(eig)': np.log(eigenvalues4[:1000])})
# Create Scatterplot
sns.lmplot(x='log(rank)', y='log(eig)', data=dataset)
reg = LinearRegression().fit(np.reshape(np.log(range(1,1001)), [-1,1]),                              np.reshape(np.log(eigenvalues4[:1000]), [-1,1]))
plt.legend(title='tanh\ny={}x+{}'.format(np.round(reg.coef_[0][0],1), np.round(reg.intercept_[0],1)),            loc='center right', frameon=False)
plt.savefig('figs/decay_tanh.png', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:




