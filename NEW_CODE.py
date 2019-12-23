import csv
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import style

power=[]

myfile=open('ALL_CHANNELS_READ.csv','r')
values=csv.reader(myfile,delimiter=',')

for cols in values:

    power.append(cols)

import numpy as np

power=np.array(power)
power=power.astype(np.float)

print(power)

no_channels=power.shape[1]
no_readings=power.shape[0]

print(no_readings)

power_x=power[:,0:no_channels-1]
power_x_bar=power[:,no_channels-1]


from sklearn.model_selection import train_test_split

train_x,test_x,train_x_bar,test_x_bar=train_test_split(power_x,power_x_bar,test_size=0.5)

for i in range(len(train_x_bar)):

    train_x_bar[i]=train_x[i][0]+train_x[i][1]+train_x[i][2]+train_x[i][3]+train_x[i][4]+5*abs(np.random.randn(1))
    test_x_bar[i]=test_x[i][0]+test_x[i][1]+test_x[i][2]+test_x[i][3]+test_x[i][4]+5*abs(np.random.rand(1))

train_x=np.reshape(train_x,(train_x.shape[1],train_x.shape[0],1))
test_x=np.reshape(test_x,(test_x.shape[1],test_x.shape[0],1))


n = 25 # Did't work for 30,40,50,100
m = 1
T = int(no_readings/2)
T1 = int(no_readings/2)
k = 5
p = 0
q = 0
alfa = 0.1
lmd = 0.1

np.random.seed(1)



#x_tot = np.zeros((T,m))
A_full = np.zeros ([k,n,m])
B_full = np.zeros ([k,T,n])
A_Hat = np.zeros ([k,n,m])

#x = np.random.rand(T, m)



error = np.zeros(20)
error_per_app = np.zeros(T*m)

##------------Functions----------------------# 

def B_Normalization (B):
    B_T = B.T
    B_T_Temp = np.random.rand(n,T)
    for i in range (n):
        B_T_Temp[i] = B_T[i]/np.linalg.norm(B_T[i])
    B_Normalized = B_T_Temp.T
    return B_Normalized

def B_Normalization_H (B):
    B_T = B.T
    B_T_Temp = np.random.rand(n*k,T)
    for i in range (n*k):
        B_T_Temp[i] = B_T[i]/np.linalg.norm(B_T[i])
    B_Normalized = B_T_Temp.T
    return B_Normalized
    

def minimizing_A (B, x):
    A = cp.Variable((n, m))
    constraints = [A >= 0]
    obj = cp.Minimize(cp.norm(x - cp.matmul(B,A),"fro") + lmd*sum(map(sum, A)))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.SCS)
    return A

def minimizing_A_Hat (B, x):
    A = cp.Variable((n*k, m))
    constraints = [A >= 0]
    obj = cp.Minimize(cp.norm(x - cp.matmul(B,A),"fro") + lmd*sum(map(sum, A)))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.SCS)
    return A

def minimizing_B (A, x):
    B = cp.Variable((T, n))
    constraints = []
    for i in range (n):
        constraints += [B >= 0,
                       (cp.norm(B[:,i])) <= 1]    
    
    obj = cp.Minimize(cp.norm(x - cp.matmul(B,A),"fro"))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.SCS)

    return B

def minimizing_B_Hat (A, x):
    B = cp.Variable((T, n*k))
    constraints = []
    for i in range (n*k):
        constraints += [B >= 0,
                       (cp.norm(B[:,i])) <= 1]    
    
    obj = cp.Minimize(cp.norm(x - cp.matmul(B,A),"fro"))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.SCS)

    return B

def plot(x_act,x_prd,g):

    global T1,m
    appl=['fridge','ac','kitchen','light','washing']

    y_act=[]
    y_prd=[]
    
    for i in range(m):
        for j in range(T):

            y_act.append(x_act[j][i])
            y_prd.append(x_prd[j][i])
    
        style.use('ggplot')
        plt.title(str(appl[g])+':house '+str(i))
        plt.xlabel('time (hours)')
        plt.ylabel('Power Consumption(Kwh)')
        plt.plot(y_act, label='Actual Readings')
        plt.plot(y_prd, label='Predicted Readings')
        plt.legend(loc=4)
        plt.savefig(str(appl[g])+'_house'+str(i)+'.png')
        plt.close()
        
        y_act=[]
        y_prd=[]


##---------Sparse coding pre-training------------#


print ("Sparse coding pre-training Started")

for j in range (k):
   
    x = train_x[j,:,:]               # x_full need to rename as x_train
    print(x)
    print(train_x_bar) 
    print (np.shape(x))
    A = np.random.rand(n, m)
    B = np.random.rand(T, n)
    print (np.shape(B))
    print (np.shape(A))
    print ("-------------------")
    
    B_Norm = B_Normalization (B)
    A_Update = minimizing_A (B_Norm,x)
    B_Update = minimizing_B (A_Update.value,x)

    print (cp.norm(x - cp.matmul(B_Update,A_Update),"fro").value)
   

    while (cp.norm(x - cp.matmul(B_Update,A_Update),"fro").value >= 0.1):

       
        A_Update = minimizing_A (B_Update.value,x)
        B_Update = minimizing_B (A_Update.value,x)
        p = p+1
               
        print('p=',p,"---- error:",cp.norm(x - cp.matmul(B_Update,A_Update),"fro").value)
    
    A_full[j,:,:] = A_Update.value
    B_full[j,:,:] = B_Update.value
    #A_Hat[j,:,:]  = A

    print ("One Aplince is Finised, No:")
    print (j)
    print ("No of While Loops")
    print (p)
    p=0
   
    del A
    del B
    

print ("Sparse coding pre-training Finished")
##----------Discriminative disaggregation training--------##
print ("Discriminative disaggregation training Started")


A_Str_V = A_full[0,:,:]
for u in range (1,k):
    A_Str_V = np.concatenate((A_Str_V, A_full[u,:,:]), axis=0)
    

B_Bar_H = B_full[0,:,:]
for v in range (1,k):
    B_Bar_H = np.concatenate((B_Bar_H, B_full[v,:,:]), axis=1)
    

x_tot = train_x_bar[:,:,0]

print (cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value )

count=0
while (cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value >= 0.1):

    Previous_Error = cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value 
    print ("2nd Iteration Started")
    A_Hat = minimizing_A_Hat (B_Bar_H, x_tot)
    B_Bar_H = B_Bar_H - alfa*(((x_tot - cp.matmul(B_Bar_H,A_Hat))*(A_Hat.value.T)) - ((x_tot - cp.matmul(B_Bar_H,A_Str_V))*(A_Str_V.T)))
    B_Bar_H = B_Normalization_H(B_Bar_H.value)
    print ("One Completed")
    print (cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value )
    Improve = Previous_Error - cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value 
    print ("Improvement:",Improve," Error:",cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value," count:",count)
    A_Hat_V = A_Str_V
    count+=1
    if Previous_Error <= cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value:
        alfa = alfa / 5
    else:
        alfa = alfa
    print('alfa:',alfa)
    
print ("Discriminative disaggregation training Finished")
##-----------Given aggregated test examples ¯X0--------##

##x_Bar = x_tot
##x_D = x_Bar + (np.random.rand(T,m)/25) ## x_D should be rename as x_bar_test

x_D = test_x_bar[:,:,0]

B_tot = np.zeros ([k,T1,n])
A_tot = np.zeros ([k,n,m])

A_Hat_D = minimizing_A_Hat(B_Bar_H, x_D)

print (np.shape (B_tot))
print (np.shape (B_full))

##for e in range (k):
##    B_tot [e,:,:] = B_Bar_H[:, (e*n) : ((e+1)*n)]

B_tot = B_full 

A_Hat_D_T = ((A_Hat_D).value).T

for f in range (k):
    A_Hat_D_T_Temp = A_Hat_D_T[:, (f*n) : ((f+1)*n)]
    A_tot [f,:,:] = A_Hat_D_T_Temp.T


##------------Import Test Data ---------------------##

            # Import x_test
            # Import x_bar_test 

## -----------Error Calculation---------------------##


for g in range (k):
    
    error[g]= cp.norm(test_x[g,:,:] - cp.matmul(B_tot [g,:,:],A_tot [g,:,:]),"fro").value     # x_full should be rename as x_test 
    print ("Error Value for:",error[g])
    plot(test_x[g,:,:],cp.matmul(B_tot [g,:,:],A_tot [g,:,:]).value,g)  # x_full should be rename as x_test 
    

