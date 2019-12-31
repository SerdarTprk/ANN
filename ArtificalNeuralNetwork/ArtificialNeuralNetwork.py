# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:19:23 2019

@author: serda
"""

import numpy as np
import sympy as sym



def DiffFonctions(a,lam,Function):
    u=sym.Symbol('u')
    """y=sym.diff(1/(1+sym.exp(-lam*u)))"""
    if Function=='Sigmoid':
        y=sym.diff(1/(1+sym.exp(-lam*u)))
        f = sym.lambdify(u, y)
        return f(a)
    elif Function=='TanH':
        y=sym.diff(sym.exp(lam*u)-sym.exp(-lam*u))/((sym.exp(lam*u)+sym.exp(-lam*u)))
        f = sym.lambdify(u, y)
        print (f(a))
        return f(a)
    elif Function=="SoftPlus":
            u=sym.Symbol('u')
            y=1/(1+sym.exp(-lam*u))
            f = sym.lambdify(u, y)
            return f(a)
def RectifiedLineer(a):
    if a<0:
        y=0
    if a>=0:
        y=a
    return y
def TanH(a,lam):
    u=sym.Symbol('u')
    y=(sym.exp(lam*u)-sym.exp(-lam*u))/((sym.exp(lam*u)+sym.exp(-lam*u)))
    f = sym.lambdify(u, y)
    return f(a)
def Lineer(u):
    return u
def Sigmoid(a,lam):
    u=sym.Symbol('u')
    y=1/(1+sym.exp(-lam*u))
    f = sym.lambdify(u, y)
    return f(a)
def BiBinary(u):
     if u<0:
        y=-1
     elif u>=0:
        y=1
     return y
def SoftPlus(u,lam):
    u=sym.Symbol('u')
    y=np.log(1+sym.exp(-lam*u))
    f = sym.lambdify(u, y)
    return f(u)
    
    

    
def updateWeight(inputs,weightIn,rule,Iteration,ActiveFonc,LearningRate,*args,**kwargs):
     lam = kwargs.get('lam', None)
     desired = kwargs.get('desired', None)

     #etc-----
     NewWeight=weightIn

##Akticasyon fonksiyonlarının çağırılması
     def Functions(u, lam):
         if ActiveFonc=='Sigmoid':
             return Sigmoid(u,lam)
         elif ActiveFonc=='BiBinary':
             return BiBinary(u)
         elif ActiveFonc=='Lineer':
             return Lineer(u)
         elif ActiveFonc=="TanH":
             return TanH(u,lam)
         elif ActiveFonc=="RectifiedLineer":
            return RectifiedLineer(u)
         elif ActiveFonc=="SoftPlus":
            return SoftPlus(u,lam)
##Belirlenen kurallara göre Delta w değerlerinin belirlenmesi
     def out(u,lam,*args,**kwargs):
         current_desired = kwargs.get('current_desired', None)
         actual=(Functions(u,lam))
         if rule=='Hebbian':
             return actual
         if rule=='Perceptron':
             return current_desired-actual
         if rule=='Delta':
             return (current_desired-actual)*DiffFonctions(u,lam,ActiveFonc)
         if rule=='Widrow-Hoff':
             return current_desired-actual
         print("Eğitim Çıkışları: ",actual)
         
         
##Eğitme kısmı       
     while (Iteration != 0):
         Iteration=Iteration-1
         inputs1=inputs 
         desired1=desired
         while inputs1.size!=0:
                  
              current_input=inputs1[:1]
              inputs1=inputs1[1:]
              if desired is None:
                  current_desired=None
              else:
                  current_desired=desired1[:1]
                  desired1=desired1[1:]
              NewWeight = np.squeeze(np.asarray(NewWeight))
              current_input = np.squeeze(np.asarray(current_input))
              
              u=np.dot(current_input,NewWeight)

              NewWeight=NewWeight+(LearningRate*out(u,lam,current_desired=current_desired)*current_input)
              print("Yeni Ağırlıklar", NewWeight)

##Girdiler-------------------------------------      

LearningRate=float(input("Öğrenme Oranını Girniz:  "))
rules=["Hebbian","Perceptron","Delta","Widrow-Hoff"]
print("Öğrenme Kuralını Seçin")
rule=input("Hebbian:0  ,  Perceptron: 1,  Delta:2,   Widrow-Hoff:3  =>  " )
rule=rules[int(rule)]
ActiveFuncs=["BiBinary","Sigmoid","TanH","RectifiedLineer","SoftPlus"]
print("Aktivasyon Fonksiyonunu seçin")

if rule!='Widrow-Hoff':
    ActiveFunc=input("BiBinary:0,  Sigmoid:1 , TanH:2,   RectifiedLineer:3 ,    SoftPlus:4=>   ")
    ActiveFunc=ActiveFuncs[int(ActiveFunc)]
else:
    ActiveFunc='Lineer'
if ActiveFunc == 'Sigmoid' and 'TanH' and 'SoftPlus' :
    lam=int(input("lambda değerini giriniz:  "))
else:
    lam=None 
    
Iteration=int(input("İterasyon sayısını girin:  "))

num_array =[]
inputs=[]
weights=[]
outputs=[]
data=input("Data sayısını girin")
num=input("Özellik sayısını girin")
for d in range(int(data)):
    for i in range(int(num)):
        n = input("num :")
        num_array.append(int(n))
    inputs.append(num_array)
    num_array=[]
print ('ARRAY: ',inputs)

print ('Ağırlıkları girin ')
for w in range(int(num)):
        n = input("num :")
        weights.append(int(n))
print ('Ağırlıklar: ',weights)


if rule !='Hebbian':
    for da in range(int(data)):
        for i in range(int("1")):
            n = input("num :")
            num_array.append(int(n))
        outputs.append(num_array)
        num_array=[]
    desired=np.array(outputs)
    print ('Çıkışlar: ',outputs)
else :
    desired=None

weightIn=np.array(weights)
inputs=np.array(inputs)
updateWeight(inputs,weightIn,rule,Iteration,ActiveFunc,LearningRate,lam=lam,desired=desired)