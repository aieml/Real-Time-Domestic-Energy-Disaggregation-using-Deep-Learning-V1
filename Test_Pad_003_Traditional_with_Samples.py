import csv
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import style

n = 25 # Did't work for 30,40,50,100
m = 1
T = 1000
T1 = 6103
k = 5
p = 0
q = 0
alfa = 0.01
lmd = 0.1

np.random.seed(1)

##-------------Importing Data---------------------#

test_status=0
train_status=0

file_names=['channel_1.csv','channel_3.csv','channel_4.csv','channel_6.csv','channel_9.csv','channel_10.csv']
files=[open(file,'r') for file in file_names]

import numpy as np

train_x=np.zeros((5,1000,1),dtype=np.float32)
train_x_bar=np.zeros((11302,1),dtype=np.float32)
test_x=np.zeros((5,6103,1),dtype=np.float32)
test_x_bar=np.zeros((6103,1),dtype=np.float32)

power_test_array=[[],[],[],[],[],[]]
power_train_array=[[],[],[],[],[],[]]

dates_test_array=[[],[],[],[],[],[]]
dates_train_array=[[],[],[],[],[],[]]

for i in range(6):

    values=csv.reader(files[i],delimiter=',')

    for cols in values:
        
        date=cols[0]
        power=eval(cols[1])

        if(date=='4/18/11 5:31 AM'):
            test_status=0
            train_status=1
        elif(date=='4/26/11 5:31 AM'):
            train_status=0
            test_status=1
        elif(date=='4/30/11 11:17 AM'):
            break

        if(train_status==1):
            power_train_array[i].append(power)
            dates_train_array[i].append(date)
            
        elif(test_status==1):
            power_test_array[i].append(power)
            dates_test_array[i].append(date)
            
    #print('+',len(power_test_array[i]))
    #print('-',len(dates_test_array[i]))
for i in range(1,6):
    
    for val in dates_test_array[i]:
        
        if(val not in dates_test_array[i]):

            index=dates_test_array[i].index(val)
            dates_test_array[i].remove(val)
            del power_test_array[i][index] 

    #print(len(power_test_array[i]),len(dates_test_array[i]))

del dates_test_array[1][3972]
del power_test_array[1][3972]
del dates_test_array[1][3972]
del power_test_array[1][3972]

del dates_test_array[2][1337]
del power_test_array[2][1337]
del dates_test_array[2][1337]
del power_test_array[2][1337]

del dates_test_array[2][3972]
del power_test_array[2][3972]
del dates_test_array[2][3972]
del power_test_array[2][3972]

del dates_test_array[3][1337]
del power_test_array[3][1337]
del dates_test_array[3][1337]
del power_test_array[3][1337]

del dates_test_array[3][3972]
del power_test_array[3][3972]
del dates_test_array[3][3972]
del power_test_array[3][3972]

del dates_test_array[4][1337]
del power_test_array[4][1337]
del dates_test_array[4][1337]
del power_test_array[4][1337]

del dates_test_array[4][3972]
del power_test_array[4][3972]
del dates_test_array[4][3972]
del power_test_array[4][3972]

del dates_test_array[5][1337]
del power_test_array[5][1337]
del dates_test_array[5][1337]
del power_test_array[5][1337]

del dates_test_array[5][3972]
del power_test_array[5][3972]
del dates_test_array[5][3972]
del power_test_array[5][3972]

del dates_train_array[1][5589]
del power_train_array[1][5589]
del dates_train_array[1][5589]
del power_train_array[1][5589]

del dates_train_array[0][6825]
del power_train_array[0][6825]

del dates_train_array[0][7920]
del power_train_array[0][7920]

del dates_train_array[1][10943]
del power_train_array[1][10943]

del dates_train_array[2][675]
del power_train_array[2][675]
del dates_train_array[2][675]
del power_train_array[2][675]

del dates_train_array[2][5589]
del power_train_array[2][5589]
del dates_train_array[2][5589]
del power_train_array[2][5589]

del dates_train_array[2][10943]
del power_train_array[2][10943]

del dates_train_array[3][675]
del dates_train_array[3][675]
del power_train_array[3][675]
del power_train_array[3][675]

del dates_train_array[3][5589]
del dates_train_array[3][5589]
del power_train_array[3][5589]
del power_train_array[3][5589]

del dates_train_array[3][9583]
del dates_train_array[3][9583]
del power_train_array[3][9583]
del power_train_array[3][9583]

del dates_train_array[3][10943]
del power_train_array[3][10943]

del dates_train_array[4][675]
del dates_train_array[4][675]
del power_train_array[4][675]
del power_train_array[4][675]

del dates_train_array[4][5589]
del dates_train_array[4][5589]
del power_train_array[4][5589]
del power_train_array[4][5589]

del dates_train_array[4][9583]
del dates_train_array[4][9583]
del power_train_array[4][9583]
del power_train_array[4][9583]

del dates_train_array[4][10943]
del power_train_array[4][10943]

del dates_train_array[5][675]
del dates_train_array[5][675]
del power_train_array[5][675]
del power_train_array[5][675]

del dates_train_array[5][5589]
del dates_train_array[5][5589]
del power_train_array[5][5589]
del power_train_array[5][5589]

del dates_train_array[5][9583]
del dates_train_array[5][9583]
del power_train_array[5][9583]
del power_train_array[5][9583]

del dates_train_array[5][10943]
del power_train_array[5][10943]


for j in range(len(power_train_array)):
    
    train_x_bar[j][0]=power_train_array[0][j]

for i in range(1,6):

    for j in range(1000):       #len(power_train_array)

        train_x[i-1][j][0]=power_train_array[i][j]



for j in range(len(power_test_array)):
    
    test_x_bar[j][0]=power_test_array[0][j]

for i in range(1,6):

    for j in range(len(power_test_array)):

        test_x[i-1][j][0]=power_test_array[i][j]




print ("Data Reading Finished")
##-------------Importing Completed------------#

x_tot = np.zeros((T,m))
A_full = np.zeros ([k,n,m])
B_full = np.zeros ([k,T,n])
A_Hat = np.zeros ([k,n,m])

x = np.random.rand(T, m)

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
    #x = train_x
    x = train_x[j,:,:]               # x_full need to rename as x_train
    A = np.random.rand(n, m)
    B = np.random.rand(T, n)

    B_Norm = B_Normalization (B)
    A_Update = minimizing_A (B_Norm,x)
    B_Update = minimizing_B (A_Update.value,x)

    print (cp.norm(x - cp.matmul(B_Update,A_Update),"fro").value)
   

    while (cp.norm(x - cp.matmul(B_Update,A_Update),"fro").value >= 0.0001):

       
        A_Update = minimizing_A (B_Update.value,x)
        B_Update = minimizing_B (A_Update.value,x)
        p = p+1
               

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
    

##A_Hat_V = A_Hat[0,:,:]
##for y in range (1,k):
##    A_Hat_V = np.concatenate((A_Hat_V, A_Hat[y,:,:]), axis=0)
    

B_Bar_H = B_full[0,:,:]
for v in range (1,k):
    B_Bar_H = np.concatenate((B_Bar_H, B_full[v,:,:]), axis=1)
    

x_tot = train_x_bar

##for w in range (k):
##    x_tot = x_tot + x_full[w,:,:]   ## x_tot shoul rename as x_bar_train 

print (cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value )

while (cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value >= 0.001):

    Previous_Error = cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value 
    print ("2nd Iteration Started")
    A_Hat = minimizing_A_Hat (B_Bar_H, x_tot)
    B_Bar_H = B_Bar_H - alfa*(((x_tot - cp.matmul(B_Bar_H,A_Hat))*(A_Hat.value.T)) - ((x_tot - cp.matmul(B_Bar_H,A_Str_V))*(A_Str_V.T)))
    B_Bar_H = B_Normalization_H(B_Bar_H.value)
    print ("One Completed")
    print (cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value )
    Improve = Previous_Error - cp.norm(x_tot - cp.matmul(B_Bar_H,A_Str_V),"fro").value 
    print ("Improvement")
    print (Improve)
    A_Hat_V = A_Str_V


print ("Discriminative disaggregation training Finished")
##-----------Given aggregated test examples ¯X0--------##

##x_Bar = x_tot
##x_D = x_Bar + (np.random.rand(T,m)/25) ## x_D should be rename as x_bar_test

x_D = test_x_bar

B_tot = np.zeros ([k,T,n])
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
    print ("Error Value for")
    #print (g)
    #print (np.round(cp.norm(x_full[g,:,:] - cp.matmul(B_tot [g,:,:],A_tot [g,:,:]),"fro").value),6) # x_full should be rename as x_test 
    #print ("-----------Differance-------------")
    #print (np.round(x_full[g,:,:] - cp.matmul(B_tot [g,:,:],A_tot [g,:,:]).value),6)
    #print ("----------Dufferance without Rounding off ---------------------------")
    #print (x_full[g,:,:] - cp.matmul(B_tot [g,:,:],A_tot [g,:,:]).value)
    #print ("-----------Original X-------------")
    #print (x_full[g,:,:])
    #print ("---------------Predicated X-------------------")
    #print (np.round(cp.matmul(B_tot [g,:,:],A_tot [g,:,:]).value))
    #print ("----------------------------------------------")
    error[g]= cp.norm(test_x[g,:,:] - cp.matmul(B_tot [g,:,:],A_tot [g,:,:]),"fro").value     # x_full should be rename as x_test 
    
    plot(test_x[g,:,:],cp.matmul(B_tot [g,:,:],A_tot [g,:,:]).value,g)  # x_full should be rename as x_test 
    
