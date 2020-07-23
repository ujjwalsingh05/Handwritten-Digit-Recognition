import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('mnist_test.csv')
data=df.values
Y=data[:,0]
X=data[:,1:]
#print(X)
#print(Y)
split=int(0.8*X.shape[0])
print(split)
print(X.shape)

X_train=X[:split,:]
Y_train=Y[:split]
X_test=X[split:,:]
Y_test=Y[split:]
print(X_test.shape)

print(Y_test[1])

def drawimage(sample):
    img=sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()

def distance(x1,x2):
    return np.sqrt((sum((x1-x2)**2)))

def knn(X,Y,queryPoint,k=10):
    vals=[]
    m=X.shape[0]
    for i in range(m):
        d=distance(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    vals=sorted(vals)
    vals=vals[:k]
    vals=np.array(vals)
    new_vals=np.unique(vals[:,1],return_counts=True)
    #print(new_vals)
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    print(int(pred))
    return (int(pred))
    
predlist=[]
split1=25

for i in range(25):
    x=knn(X_train,Y_train,X_test[i])
    predlist.append(x)
    

Y_tst=Y_test[:25]    
    
score=float((predlist==Y_tst).sum()/split1)*100
print(score)
    
    
    
    

    
     


