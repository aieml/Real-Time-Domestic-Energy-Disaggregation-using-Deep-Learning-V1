import csv
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import style

n = 25 # Did't work for 30,40,50,100
m = 5
T = 24
k = 5
p = 0
q = 0
alfa = 0.00001
lmd = 0.1

np.random.seed(1)

##-------------Importing Data---------------------#

x_full = np.zeros((k,T,m),dtype=np.float32)

fridge=open('Fridge.csv')
ac=open('AC.csv')
kitchen=open('Kitchen.csv')
light=open('Light.csv')
washing=open('Washing.csv')

files=[fridge,ac,kitchen,light,washing]

for d in range(k):

    values=csv.reader(files[d],delimiter=',')
    b=0
    
    for cols in values:
        
        for c in range(m):

            x_full[d][b][c]=eval(cols[c])/10

        b=b+1


print ("Data Reading Finished")
##-------------Importing Completed------------#

#x_full = np.random.rand (k,T,m)
x_tot = np.zeros((T,m))
A_full = np.zeros ([k,n,m])
B_full = np.zeros ([k,T,n])
A_Hat = np.zeros ([k,n,m])

x = np.random.rand(T, m)

error = np.zeros(20)
error_per_app = np.zeros(T*m)

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
    prob.solve()
    return A

def minimizing_A_Hat (B, x):
    A = cp.Variable((n*k, m))
    constraints = [A >= 0]
    obj = cp.Minimize(cp.norm(x - cp.matmul(B,A),"fro") + lmd*sum(map(sum, A)))
    prob = cp.Problem(obj,constraints)
    prob.solve()
    return A

def minimizing_B (A, x):
    B = cp.Variable((T, n))
    constraints = []
    for i in range (n):
        constraints += [B >= 0,
                       (cp.norm(B[:,i])) <= 1]    
    
    obj = cp.Minimize(cp.norm(x - cp.matmul(B,A),"fro"))
    prob = cp.Problem(obj,constraints)
    prob.solve()

    return B

def minimizing_B_Hat (A, x):
    B = cp.Variable((T, n*k))
    constraints = []
    for i in range (n*k):
        constraints += [B >= 0,
                       (cp.norm(B[:,i])) <= 1]    
    
    obj = cp.Minimize(cp.norm(x - cp.matmul(B,A),"fro"))
    prob = cp.Problem(obj,constraints)
    prob.solve()

    return B

##---------Sparse coding pre-training------------#
print ("Sparse coding pre-training Started")



for j in range (k):
    x = x_full[j,:,:]
    A = np.random.rand(n, m)
    B = np.random.rand(T, n)

    B_Norm = B_Normalization (B)
    A_Update = minimizing_A (B_Norm,x)
    B_Update = minimizing_B (A_Update.value,x)

    print (cp.norm(x - cp.matmul(B_Update,A_Update),"fro").value)
   

    while (cp.norm(x - cp.matmul(B_Update,A_Update),"fro").value >= 0.001):

       
        A_Update = minimizing_A (B_Update.value,x)
        B_Update = minimizing_B (A_Update.value,x)
        p = p+1
               

    A_full[j,:,:] = A_Update.value
    B_full[j,:,:] = B_Update.value
    A_Hat[j,:,:]  = A 

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
    

A_Hat_V = A_Hat[0,:,:]
for y in range (1,k):
    A_Hat_V = np.concatenate((A_Hat_V, A_Hat[y,:,:]), axis=0)
    

B_Bar_H = B_full[0,:,:]
for v in range (1,k):
    B_Bar_H = np.concatenate((B_Bar_H, B_full[v,:,:]), axis=1)
    

for w in range (k):
    x_tot = x_tot + x_full[w,:,:]

while (cp.norm(x_tot - cp.matmul(B_Bar_H,A_Hat_V),"fro").value >= 0.0011):
    print ("2nd Iteration Started")
    A_Hat = minimizing_A_Hat (B_Bar_H, x_tot)
    B_Bar_H = B_Bar_H - alfa*(((x_tot - cp.matmul(B_Bar_H,A_Hat))*(A_Hat.value.T)) - ((x_tot - cp.matmul(B_Bar_H,A_Str_V))*(A_Str_V.T)))
    B_Bar_H = B_Normalization_H(B_Bar_H.value)
    print ("One Completed")
    print (cp.norm(x_tot - cp.matmul(B_Bar_H,A_Hat),"fro").value )
    A_Hat_V = A_Hat


print ("Discriminative disaggregation training Finished")
##-----------Given aggregated test examples ¯X0--------##

x_Bar = x_tot
x_D = x_Bar + (np.random.rand(T,m)/10)
B_tot = np.zeros ([k,T,n])
A_tot = np.zeros ([k,n,m])

A_Hat_D = minimizing_A_Hat(B_Bar_H, x_D)

for e in range (k):
    B_tot [e,:,:] = B_Bar_H[:, (e*n) : ((e+1)*n)]


A_Hat_D_T = ((A_Hat_D).value).T

for f in range (k):
    A_Hat_D_T_Temp = A_Hat_D_T[:, (f*n) : ((f+1)*n)]
    A_tot [f,:,:] = A_Hat_D_T_Temp.T


## -----------Error Calculation---------------------##


x_full_E = np.zeros ((T,m))
B_tot_E = np.zeros ((T,n))
A_tot_E = np.zeros ((n,m))

x_full_E = x_full[0,:,:]
B_tot_E = B_tot [0,:,:]
A_tot_E = A_tot [0,:,:]



x_E = x_full_E - cp.matmul(B_tot_E,A_tot_E).value

print ("Error Plotting")
print (np.shape(x_E))
print ("x_E")
print (x_E)
print ("x_full[0,:,:]")
print (x_full[0,:,:])
print ("cp.matmul(B_tot_E,A_tot_E).value")
print (cp.matmul(B_tot_E,A_tot_E).value)

def plot(x_act,x_prd,g):

    global T,m
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

for g in range (k):
    print ("Error Value for")
    print (g)
    print (np.round(cp.norm(x_full[g,:,:] - cp.matmul(B_tot [g,:,:],A_tot [g,:,:]),"fro").value),6)
    print ("-----------Differance-------------")
    print (np.round(x_full[g,:,:] - cp.matmul(B_tot [g,:,:],A_tot [g,:,:]).value),6)
    print ("----------Dufferance without Rounding off ---------------------------")
    print (x_full[g,:,:] - cp.matmul(B_tot [g,:,:],A_tot [g,:,:]).value)
    #print ("-----------Original X-------------")
    #print (x_full[g,:,:])
    #print ("---------------Predicated X-------------------")
    #print (np.round(cp.matmul(B_tot [g,:,:],A_tot [g,:,:]).value))
    #print ("----------------------------------------------")
    error[g]= cp.norm(x_full[g,:,:] - cp.matmul(B_tot [g,:,:],A_tot [g,:,:]),"fro").value
    
    plot(x_full[g,:,:],cp.matmul(B_tot [g,:,:],A_tot [g,:,:]).value,g)
    
    
