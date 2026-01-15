import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
import requests
import base64
from scipy.ndimage import label
import skimage.morphology
import skimage.measure
from skimage.measure import regionprops, label
import math
import random as rnd

class RSA:
    def __init__(self, image, N, T, lb, ub, dim):
        self.N = N
        self.T = T
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.image = image

    def initialization(self):
        X = np.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb
        return X

    def objective_function(self,thresholds):
        # Histogram of the image
        histogram, _ = np.histogram(self.image, bins=256, range=(0, 256), density=True)
        thresholds = [int(round(t)) for t in thresholds]
        thresholds = sorted(thresholds)

        p_regions = []
        w_regions = []
        A_regions = []
        J = 0

        for region_idx in range(len(thresholds) + 1):
            start = 0 if region_idx == 0 else thresholds[region_idx - 1]
            end = thresholds[region_idx] if region_idx < len(thresholds) else 255

            p_region = [histogram[i] / np.sum(histogram) for i in range(start, end)]
            p_regions.append(p_region)
            w_regions.append(np.sum(p_region))

            A_region = [-(p / w_regions[region_idx]) * np.log(p / w_regions[region_idx]) if p > 0 else 0.001 for p in p_region]
            A_regions.append(A_region)

            J += np.sum(A_region)

        return J


    def resolve(self):
        #print('RSA está trabajando en tu problema')
        Best_P = np.zeros(self.dim)  # best positions
        Best_F = 0.0  # best fitness

        X = self.initialization()  # Initialize the positions of solution
        Xnew = np.zeros((self.N, self.dim))
        Conv = np.zeros(self.T)  # Convergence array

        t = 1  # starting iteration
        Alpha = 0.1  # the best value 0.1
        Beta = 0.1  # the best value 0.005

        Ffun = np.zeros(X.shape[0])  # (old fitness values)
        Ffun_new = np.zeros(X.shape[0])  # (new fitness values)

        for i in range(X.shape[0]):
            Ffun[i] = self.objective_function(X[i, :])  # Calculate the fitness values of solutions
            if Ffun[i] > Best_F:
                Best_F = Ffun[i]
                Best_P = X[i, :]

        while t < self.T + 1:  # Main loop
            ES = 2 * np.random.randn() * (1 - (t / self.T))  # Probability Ratio

            for i in range(X.shape[0]):
                R = (Best_P - X[np.random.randint(0, X.shape[0])]) / ((Best_P) + np.finfo(float).eps)
                P = Alpha + (X[i, :] - np.mean(X[i, :])) / ((Best_P) * (self.ub - self.lb) + np.finfo(float).eps)
                Eta = Best_P * P

                if t < self.T / 4:
                    Xnew[i, :] = Best_P - Eta * Beta - R * np.random.rand()
                elif self.T / 4 <= t < 2 * self.T / 4:
                    Xnew[i, :] = Best_P * X[np.random.randint(0, X.shape[0])] * ES * np.random.rand()
                elif 2 * self.T / 4 <= t < 3 * self.T / 4:
                    Xnew[i, :] = Best_P * P * np.random.rand()
                else:
                    Xnew[i, :] = Best_P - Eta * np.finfo(float).eps - R * np.random.rand()

                Flag_UB = Xnew[i, :] > self.ub  # check if they exceed (up) the boundaries
                Flag_LB = Xnew[i, :] < self.lb  # check if they exceed (down) the boundaries
                Xnew[i, :] = (Xnew[i, :] * (~(Flag_UB + Flag_LB))) + self.ub * Flag_UB + self.lb * Flag_LB
                Ffun_new[i] = self.objective_function(Xnew[i, :])

                if Ffun_new[i] > Ffun[i]:
                    X[i, :] = Xnew[i, :]
                    Ffun[i] = Ffun_new[i]
                if Ffun[i] > Best_F:
                    Best_F = Ffun[i]
                    Best_P = X[i, :]

            Conv[t-1] = Best_F  # Update the convergence curve

            #if t % 50 == 0:  # Print the best solution details after every 50 iterations
                #print(f"En la iteración {t}, la mejor solución tiene una aptitud de {Best_F}")
            t += 1

        return Best_F, Best_P, Conv

class TSO:
    def __init__(self, image, Particles_no, Max_iter, Low, Up, Dim):
        self.Particles_no = Particles_no
        self.Max_iter = Max_iter
        self.Low = Low
        self.Up = Up
        self.Dim = Dim
        self.image = image
        
    def initialization(self):
        X = np.random.rand(self.Particles_no, self.Dim) * (self.Up - self.Low) + self.Up
        return X

    def objective_function(self,thresholds):
        # Histogram of the image
        histogram, _ = np.histogram(self.image, bins=256, range=(0, 255), density=True)
        thresholds = [int(round(t)) for t in thresholds]
        thresholds = sorted(thresholds)

        p_regions = []
        w_regions = []
        A_regions = []
        J = 0

        for region_idx in range(len(thresholds) + 1):
            start = 0 if region_idx == 0 else thresholds[region_idx - 1]
            end = thresholds[region_idx] if region_idx < len(thresholds) else 255

            p_region = [histogram[i] / np.sum(histogram) for i in range(start, end)]
            p_regions.append(p_region)
            w_regions.append(np.sum(p_region))

            A_region = [-(p / w_regions[region_idx]) * np.log(p / w_regions[region_idx]) if p > 0 else 0.001 for p in p_region]
            A_regions.append(A_region)

            J += np.sum(A_region)

        return J
    
    def resolve(self):
        #print('TSO está trabajando en tu problema')
        Tuna1 = np.zeros(self.Dim)
        Tuna1_fit = float(0)
        T = self.initialization()
        Iter = 0
        aa = 0.7
        z = 0.05
        fitness = np.zeros(self.Particles_no)
        fit_old = np.zeros(self.Particles_no)
        C_old = np.zeros((self.Particles_no, self.Dim))
        while Iter < self.Max_iter:
            C = Iter / self.Max_iter
            a1 = aa + (1 - aa) * C
            a2 = (1 - aa) - (1 - aa) * C
            for i in range(T.shape[0]):
                Flag4ub = T[i, :] > self.Up
                Flag4lb = T[i, :] < self.Low
                T[i, :] = (T[i, :] * (~(Flag4ub + Flag4lb))) + self.Up * Flag4ub + self.Low * Flag4lb
                fitness[i] = self.objective_function(T[i, :])
                if fitness[i] > Tuna1_fit:
                    Tuna1_fit = fitness[i]
                    Tuna1 = T[i, :]

            if Iter == 0:
                fit_old = np.copy(fitness)
                C_old = np.copy(T)

            for i in range(self.Particles_no):
                if fit_old[i] > fitness[i]:
                    fitness[i] = fit_old[i]
                    T[i, :] = C_old[i, :]

            C_old = np.copy(T)
            fit_old = np.copy(fitness)
            t = (1 - Iter / self.Max_iter) ** (Iter / self.Max_iter)
            if np.random.rand() < z:
                T[0, :] = (self.Up - self.Low) * np.random.rand(self.Dim) + self.Low
            else:
                if 0.5 < np.random.rand():
                    r1 = np.random.rand()
                    Beta = np.exp(r1 * np.exp(3 * np.cos(np.pi * ((self.Max_iter - Iter + 1) / self.Max_iter)))) * (
                            np.cos(2 * np.pi * r1))
                    if C > np.random.rand():
                        T[0, :] = a1 * (Tuna1 + Beta * np.abs(Tuna1 - T[0, :])) + a2 * T[0, :]
                    else:
                        IndivRand = np.random.rand(self.Dim) * (self.Up - self.Low) + self.Low
                        T[0, :] = a1 * (IndivRand + Beta * np.abs(IndivRand - T[i, :])) + a2 * T[0, :]
                else:
                    TF = (np.random.rand() > 0.5) * 2 - 1
                    if 0.5 > np.random.rand():
                        T[0, :] = Tuna1 + np.random.rand(self.Dim) * (Tuna1 - T[0, :]) + TF * t ** 2 * (Tuna1 - T[0, :])
                    else:
                        T[0, :] = TF * t ** 2 * T[0, :]
            for i in range(1, self.Particles_no):
                if np.random.rand() < z:
                    T[i, :] = (self.Up - self.Low) * np.random.rand(self.Dim) + self.Low
                else:
                    if 0.5 < np.random.rand():
                        r1 = np.random.rand()
                        Beta = np.exp(r1 * np.exp(3 * np.cos(np.pi * ((self.Max_iter - Iter + 1) / self.Max_iter)))) * (
                                np.cos(2 * np.pi * r1))
                        if C > np.random.rand():
                            T[i, :] = a1 * (Tuna1 + Beta * np.abs(Tuna1 - T[i, :])) + a2 * T[i - 1, :]
                        else:
                            IndivRand = np.random.rand(self.Dim) * (self.Up - self.Low) + self.Low
                            T[i, :] = a1 * (IndivRand + Beta * np.abs(IndivRand - T[i, :])) + a2 * T[i - 1, :]
                    else:
                        TF = (np.random.rand() > 0.5) * 2 - 1
                        if 0.5 > np.random.rand():
                            T[i, :] = Tuna1 + np.random.rand(self.Dim) * (Tuna1 - T[i, :]) + TF * t ** 2 * (Tuna1 - T[i, :])
                        else:
                            T[i, :] = TF * t ** 2 * T[i, :]

            Iter += 1
            #if Iter % 50 == 0:  # Print the best solution details after every 50 iterations
                #print(f"En la iteración {Iter}, la mejor solución tiene una aptitud de {Tuna1_fit}")
        Convergence_curve = np.zeros(Iter)
        Convergence_curve[:Iter] = Tuna1_fit
        return Tuna1_fit, Tuna1, Convergence_curve

class HBA:
    def __init__(self, image, N, T, lb, ub,dim):
        self.image = image
        self.N = N
        self.T = T
        self.ub = ub  # Added the upper bound as an instance variable
        self.lb = lb
        self.dim = dim

    def initialization(self):
        if self.ub.shape[0] == 1:
            X = np.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb
        else:
            X = np.zeros((self.N, self.dim))
            for i in range(self.dim):
                high = self.ub[i]
                low = self.lb[i]
                X[:, i] = np.random.rand(self.N) * (high - low) + low
        return X

    def objective_function(self,thresholds):
        # Histogram of the image
        histogram, _ = np.histogram(self.image, bins=256, range=(0, 256), density=True)
        thresholds = [int(round(t)) for t in thresholds]
        thresholds = sorted(thresholds)

        p_regions = []
        w_regions = []
        A_regions = []
        J = 0

        for region_idx in range(len(thresholds) + 1):
            start = 0 if region_idx == 0 else thresholds[region_idx - 1]
            end = thresholds[region_idx] if region_idx < len(thresholds) else 255

            p_region = [histogram[i] / np.sum(histogram) for i in range(start, end)]
            p_regions.append(p_region)
            w_regions.append(np.sum(p_region))

            A_region = [-(p / w_regions[region_idx]) * np.log(p / w_regions[region_idx]) if p > 0 else 0.001 for p in p_region]
            A_regions.append(A_region)

            J += np.sum(A_region)

        return J

    def check_constraint(self, thresholds):
        if thresholds[0] != thresholds[1]:
            return True
        else:
            return False

    def fun_calcobjfunc(self, X):
        N = X.shape[0]
        Y = np.zeros(N)
        for i in range(N):
            Y[i] = self.objective_function(X[i, :])
        return Y

    def Intensity(self, Xprey, X):
        di = np.zeros(self.N)
        S = np.zeros(self.N)
        for i in range(self.N - 1):
            di[i] = np.linalg.norm((X[i, :] - Xprey + np.finfo(float).eps)) ** 2
            S[i] = np.linalg.norm((X[i, :] - X[i + 1, :] + np.finfo(float).eps)) ** 2
        di[self.N - 1] = np.linalg.norm((X[self.N - 1, :] - Xprey + np.finfo(float).eps)) ** 2
        S[self.N - 1] = np.linalg.norm((X[self.N - 1, :] - X[0, :] + np.finfo(float).eps)) ** 2
        I = np.zeros(self.N)
        for i in range(self.N):
            r2 = np.random.random()
            I[i] = r2 * S[i] / (4 * np.pi * di[i])
        return I

    def resolve(self):
        #print('HBASegmenter is working on your problem')

        beta = 6
        C = 2
        vec_flag = [1, -1]

        Best_P = np.zeros(self.dim)
        Best_F = float(0)
        X = self.initialization()
        Xnew = np.zeros((self.N, self.dim))
        Conv = np.zeros(T)
        Ffun = np.zeros(X.shape[0])
        Ffun_new = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            X[i, :] = [int(x) for x in X[i, :]]

            Ffun[i] = self.objective_function(X[i, :])
            if Ffun[i] > Best_F:
                Best_F = Ffun[i]
                Best_P = X[i, :]

        for t in range(T):
            alpha = C * np.exp(-t / T)
            I = self.Intensity(Best_P, X)

            Xnew = np.zeros((self.N, self.dim))

            for i in range(self.N):
                r = np.random.random()
                F = vec_flag[int(np.floor(2 * np.random.random()))]
                di = (Best_P - X[i, :])
                if r < 0.5:
                    r3 = np.random.random()
                    r4 = np.random.random()
                    r5 = np.random.random()
                    Xnew[i, :] = Best_P + F * beta * I[i] * Best_P + F * r3 * alpha * (di) * np.abs(
                        np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))
                else:
                    r7 = np.random.random()
                    Xnew[i, :] = Best_P + F * r7 * alpha * di

                FU = Xnew[i, :] > ub
                FL = Xnew[i, :] < lb
                Xnew[i, :] = (Xnew[i, :] * (~(FU + FL))) + ub * FU + lb * FL

                Ffun_new[i] = self.objective_function(Xnew[i, :])
                if Ffun_new[i] > Ffun[i]:
                    X[i, :] = Xnew[i, :]
                    Ffun[i] = Ffun_new[i]
                if Ffun[i] > Best_F:
                    Best_F = Ffun[i]
                    Best_P = X[i, :]

            Conv[t] = Best_F
            #if t % 50 == 0:
                #print(f"At iteration {t}, the best solution fitness is {Best_F}")

        return Best_F, Best_P, Conv

class OPA:
    def __init__(self, image,N,tmax, lb, ub, dim):
        self.image = image
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.tmax = tmax
        self.N = N

    def initialization(self):
        X = np.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb
        return X

    def objective_function(self,thresholds):
        # Histogram of the image
        histogram, _ = np.histogram(self.image, bins=256, range=(0, 256), density=True)
        thresholds = [int(round(t)) for t in thresholds]
        thresholds = sorted(thresholds)

        p_regions = []
        w_regions = []
        A_regions = []
        J = 0

        for region_idx in range(len(thresholds) + 1):
            start = 0 if region_idx == 0 else thresholds[region_idx - 1]
            end = thresholds[region_idx] if region_idx < len(thresholds) else 255

            p_region = [histogram[i] / np.sum(histogram) for i in range(start, end)]
            p_regions.append(p_region)
            w_regions.append(np.sum(p_region))

            A_region = [-(p / w_regions[region_idx]) * np.log(p / w_regions[region_idx]) if p > 0 else 0.001 for p in p_region]
            A_regions.append(A_region)

            J += np.sum(A_region)

        return J


    def choose(self, orcas):
        while True:
            oj = np.random.randint(len(orcas))
            ok = np.random.randint(len(orcas))
            ol = np.random.randint(len(orcas))
            if oj != ok and oj != ol and ok != ol:
                break
        return orcas[oj], orcas[ok], orcas[ol]

    def resolve(self):
        #print('OPA está trabajando en tu problema')

        p = 0.5
        q = 0.75
        F = 2
        Best_P = np.zeros(self.dim)
        Best_F = float(0)
        tmax = self.tmax

        X = self.initialization()
        Xnew = np.zeros((self.N, self.dim))
        Conv = np.zeros(self.tmax)

        Ffun = np.zeros(self.N)
        Ffun_new = np.zeros(self.N)

        for i in range(self.N):
            Ffun[i] = self.objective_function(X[i, :])
            if Ffun[i] > Best_F:
                Best_F = Ffun[i]
                Best_P = X[i, :]

        for t in range(tmax):
            Xnew = np.zeros((self.N, self.dim))  # Notar el cambio aquí
            u = 2 * (np.random.rand() - 0.5) * (tmax - t) / tmax
            M = np.mean(X)
            for i in range(self.N):  # Notar el cambio aquí
                oj, ok, ol = self.choose(X)
                dimension = X[i, :].shape[0]
                if p > np.random.rand():
                    if q > np.random.rand():
                        for k in range(dimension):
                            a = np.random.rand()
                            b = np.random.rand()
                            c = 1 - b
                            d = np.random.rand()
                            velocity = a * (d * Best_P - F * (b * M + c * X[k,:]))
                    else:
                        for k in range(dimension):
                            e = np.random.rand()
                            velocity = e * Best_P - X[k,:]
                else:
                    for k in range(dimension):
                        Xnew[k,:] = oj + u * (ok - ol)

                Flag_UB = Xnew[i, :] > ub  # check if they exceed (up) the boundaries
                Flag_LB = Xnew[i, :] < lb  # check if they exceed (down) the boundaries
                Xnew[i, :] = (Xnew[i, :] * (~(Flag_UB + Flag_LB))) + ub * Flag_UB + lb * Flag_LB

                Ffun_new[i] = self.objective_function(Xnew[i, :])
                if Ffun_new[i] > Ffun[i]:
                    X[i, :] = Xnew[i, :]
                    Ffun[i] = Ffun_new[i]
                if Ffun[i] > Best_F:
                    Best_F = Ffun[i]
                    Best_P = X[i, :]

            Conv[t] = Best_F
            #if t % 50 == 0:  # Print the best universe details after every 50 iterations
                #print(f"At iteration {t}, the best solution fitness is {Best_F}")


        return Best_F, Best_P, Conv


    
class BES:
    def __init__(self, image,N,tmax, lb, ub, dim):
        self.N = N
        self.T = T
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.image = image



    def objective_function(self,thresholds):
        # Histogram of the image
        histogram, _ = np.histogram(self.image, bins=256, range=(0, 256), density=True)
        thresholds = [int(round(t)) for t in thresholds]
        thresholds = sorted(thresholds)

        p_regions = []
        w_regions = []
        A_regions = []
        J = 0

        for region_idx in range(len(thresholds) + 1):
            start = 0 if region_idx == 0 else thresholds[region_idx - 1]
            end = thresholds[region_idx] if region_idx < len(thresholds) else 255

            p_region = [histogram[i] / np.sum(histogram) for i in range(start, end)]
            p_regions.append(p_region)
            w_regions.append(np.sum(p_region))

            A_region = [-(p / w_regions[region_idx]) * np.log(p / w_regions[region_idx]) if p > 0 else 0.001 for p in p_region]
            A_regions.append(A_region)

            J += np.sum(A_region)

        return J
    def population(self,pop_size,nVars):
        X = np.random.rand(pop_size, nVars) * (self.ub - self.lb) + self.lb
        return X

    def create_x_y_x1_y1__(self,pop_size,a_factor,R_factor):
        phi = a_factor * np.pi * np.random.uniform(0, 1, pop_size)
        r = phi + R_factor * np.random.uniform(0, 1, pop_size)
        xr, yr = r * np.sin(phi), r * np.cos(phi)
        r1 = phi1 = a_factor * np.pi * np.random.uniform(0, 1, pop_size)
        xr1, yr1 = r1 * np.sinh(phi1), r1 * np.cosh(phi1)
        x_list = xr / np.max(xr)
        y_list = yr / np.max(yr)
        x1_list = xr1 / np.max(xr1)
        y1_list = yr1 / np.max(yr1)
        return x_list, y_list, x1_list, y1_list
    def resolve(self):
        N = self.N
        T = self.T
        lb = self.lb
        ub = self.ub
        dim = self.dim
        #print('BES is working on your problem...')
        nVars=dim
        pop_size=N
        t = 0
        a_factor=10
        R_factor=1.5
        alpha=2
        c1=2
        c2=2
        P_best = np.zeros(nVars)
        P_g = float(0)
        P_new = np.zeros(nVars)
        x=self.population(pop_size,nVars)
        conv=[]
        for i in x:
            Flag_UB = i > ub  # check if they exceed (up) the boundaries
            Flag_LB = i < lb  # check if they exceed (down) the boundaries
            i= (i * (~(Flag_UB + Flag_LB))) + ub * Flag_UB + lb * Flag_LB
            Ffun = self.objective_function(i)
            if(Ffun>P_g):
                P_best=i
                P_g=Ffun
        while (t<T):
            x_list, y_list, x1_list, y1_list = self.create_x_y_x1_y1__(pop_size,a_factor,R_factor)
            P_new = np.zeros(nVars)
            Ffun= self.objective_function(P_best)
            #selección de espacio
            for i in x:
                for j in range(0,i.size,1):
                    P_new[j] = P_best[j] + alpha*np.random.rand()*(np.mean(i)-i[j])     
                Flag_UB = P_new> ub  # check if they exceed (up) the boundaries
                Flag_LB = P_new < lb  # check if they exceed (down) the boundaries
                P_new= (P_new * (~(Flag_UB + Flag_LB))) + ub * Flag_UB + lb * Flag_LB
                Ffun_new=self.objective_function(P_new)

                if(Ffun_new>Ffun):
                    P_best=P_new    
                Flag_UB = i > ub  # check if they exceed (up) the boundaries
                Flag_LB = i < lb  # check if they exceed (down) the boundaries
                i= (i * (~(Flag_UB + Flag_LB))) + ub * Flag_UB + lb * Flag_LB
                Ffun_i= self.objective_function(i)
                if(Ffun_i>Ffun_new):
                    P_best=i
            Flag_UB = P_best > ub  # check if they exceed (up) the boundaries
            Flag_LB = P_best < lb  # check if they exceed (down) the boundaries
            P_best= (P_best * (~(Flag_UB + Flag_LB))) + ub * Flag_UB + lb * Flag_LB
            Ffun= self.objective_function(P_best)
            #búsqueda en el espacio
            for i in range(0,pop_size,1):
                P_new = np.zeros(nVars)
                #generamos valores aleatorios por el tamaño de pop size
                for j in range(0,P_best.size,1):
                    if (j!=len(P_best)-1):
                        P_new[j]=P_best[j]+y_list[i]*(P_best[j]-P_best[j+1])+x_list[i]*(P_best[j]-np.mean(P_best))
                    else:
                        P_new[j]=P_best[j]+y_list[i]*(P_best[j]-P_best[-1])+x_list[i]*(P_best[j]-np.mean(P_best))
                Flag_UB = P_new > ub  # check if they exceed (up) the boundaries
                Flag_LB = P_new < lb  # check if they exceed (down) the boundaries
                P_new = (P_new * (~(Flag_UB + Flag_LB))) + ub * Flag_UB + lb * Flag_LB
                Ffun_new= self.objective_function(P_new)
                if(Ffun_new>Ffun):
                    P_best=P_new
            Ffun= self.objective_function(P_best)
            #swoop en el espacio
            for i in range(0,pop_size,1):
                P_new = np.zeros(nVars)
                for j in range(0,P_best.size,1):
                    P_new[j]=np.random.uniform(0,1)*(P_best[j])+x1_list[i]*(P_best[j]-c1*np.mean(P_best))+y1_list[i]*(P_best[j]-c2*(P_best[j]))
                Flag_UB = P_new > ub  # check if they exceed (up) the boundaries
                Flag_LB = P_new < lb  # check if they exceed (down) the boundaries
                P_new = (P_new * (~(Flag_UB + Flag_LB))) + ub * Flag_UB + lb * Flag_LB
                Ffun_new= self.objective_function(P_new)
                if(Ffun_new>Ffun):
                    P_best=P_new
            Flag_UB = P_best > ub  # check if they exceed (up) the boundaries
            Flag_LB = P_best < lb  # check if they exceed (down) the boundaries
            P_best = (P_best * (~(Flag_UB + Flag_LB))) + ub * Flag_UB + lb * Flag_LB  
            Ffun= self.objective_function(P_best)
            if(Ffun>P_g):
                P_g=self.objective_function(P_best)
            conv.append(P_g)
            t=t+1
            #if t % 50 == 0:  # Print the best universe details after every 50 iterations
                #print(f"At iteration {t}, the best solution fitness is {P_g}")
        return P_g,P_best,conv
    
class GWO:
    def __init__(self, image, N=30, T=100, lb=np.array([0]), ub=np.array([255]), dim=7):  # Nota el cambio de dim a 7
        self.image = image
        self.histogram, _ = np.histogram(self.image.flatten(), 256, [0,256])  # Calculamos el histograma
        self.N = N
        self.T = T
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.wolves = self.init_wolves()
        self.conv = []

    def clip(self, position):
        """Garantizar que la posición esté dentro de los límites."""
        return [np.clip(position[i], self.lb[i], self.ub[i]) for i in range(self.dim)]

    def move_wolf(self, wolf, alpha, beta, delta, a):
        new_position = []
        for j in range(self.dim):
            A = 2 * a * rnd.random() - a
            new_threshold = wolf[j] + A * (alpha[j] - wolf[j])
            new_position.append(new_threshold)
        new_position = self.clip(new_position)  # Clip the position to the bounds
        return new_position if self.objective_function(new_position) > self.objective_function(wolf) else wolf

    def objective_function(self,thresholds):
        # Histogram of the image
        histogram, _ = np.histogram(self.image, bins=256, range=(0, 256), density=True)
        thresholds = [int(round(t)) for t in thresholds]
        thresholds = sorted(thresholds)

        p_regions = []
        w_regions = []
        A_regions = []
        J = 0

        for region_idx in range(len(thresholds) + 1):
            start = 0 if region_idx == 0 else thresholds[region_idx - 1]
            end = thresholds[region_idx] if region_idx < len(thresholds) else 255

            p_region = [histogram[i] / np.sum(histogram) for i in range(start, end)]
            p_regions.append(p_region)
            w_regions.append(np.sum(p_region))

            A_region = [-(p / w_regions[region_idx]) * np.log(p / w_regions[region_idx]) if p > 0 else 0.001 for p in p_region]
            A_regions.append(A_region)

            J += np.sum(A_region)

        return J


    def init_wolves(self):
        wolves = []
        for _ in range(self.N):
            position = [np.random.uniform(self.lb[i], self.ub[i]) for i in range(self.dim)]
            wolves.append(position)
        return wolves


    def update_alpha_beta_delta(self):
        sorted_wolves = sorted(self.wolves, key=self.objective_function, reverse=True)
        return sorted_wolves[0], sorted_wolves[1], sorted_wolves[2]


    def evolve(self):
        t = 1
        while t <= self.T:
            a = 2 * (1 - t / self.T)
            alpha, beta, delta = self.update_alpha_beta_delta()

            for i in range(self.N):
                self.wolves[i] = self.move_wolf(self.wolves[i], alpha, beta, delta, a)
            
            # Añadimos el mejor fitness a la convergencia
            best_fitness = self.objective_function(alpha)
            self.conv.append(best_fitness)
            #if t % 50 == 0:  # Print the best solution details after every 50 iterations
                #print(f"En la iteración {t}, la mejor solución tiene una aptitud de {best_fitness}")           
            t += 1

    def solve(self):
        #print('GWO está trabajando en tu problema')
        self.evolve()
        best_wolf = sorted(self.wolves, key=self.objective_function, reverse=True)[0]
        best_threshold = best_wolf[0]
        _, segmented_image = cv2.threshold(self.image, best_threshold, 255, cv2.THRESH_BINARY)

        # Return the best fitness, best position, and convergence (we haven't defined convergence yet)
        # So, for now, let's just return None for convergence.
        return self.objective_function(best_wolf), best_wolf, self.conv
    
#   hho = HHO(image, lb, ub, dim, N, T)
class HHO:
    def __init__(self, image, N, T, lb, ub, dim):
        self.image = image
        self.N = N  # Number of search agents (previously SearchAgents_no)
        self.T = T  # Max number of iterations (previously Max_iter)
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.conv = []


    def objective_function(self, thresholds):
        # Histogram of the image
        histogram, _ = np.histogram(self.image, bins=256, range=(0, 256), density=True)
        thresholds = [int(round(t)) for t in thresholds]
        thresholds = sorted(thresholds)

        p_regions = []
        w_regions = []
        A_regions = []
        J = 0

        for region_idx in range(len(thresholds) + 1):
            start = 0 if region_idx == 0 else thresholds[region_idx - 1]
            end = thresholds[region_idx] if region_idx < len(thresholds) else 255

            p_region = [histogram[i] / np.sum(histogram) for i in range(start, end)]
            p_regions.append(p_region)
            w_regions.append(np.sum(p_region))

            A_region = [-(p / w_regions[region_idx]) * np.log(p / w_regions[region_idx]) if p > 0 else 0.001 for p in p_region]
            A_regions.append(A_region)

            J += np.sum(A_region)

        return J
    def optimize(self):
            Rabbit_Location = np.zeros(self.dim)
            Rabbit_Energy = float(0)
            

            if not isinstance(self.lb, np.ndarray):
                self.lb = np.full(self.dim, self.lb)
                self.ub = np.full(self.dim, self.ub)
            self.lb = np.asarray(self.lb)
            self.ub = np.asarray(self.ub)

            X = np.asarray([(x * (self.ub - self.lb) + self.lb) for x in np.random.uniform(0, 1, (self.N, self.dim))])

            convergence_curve = np.zeros(self.T)

            #print('HHO está trabajando en tu problema')

            t = 0

            while t < self.T:
                for i in range(0, self.N):
                    # Check boundries

                    X[i,:]=np.clip(X[i,:], lb, ub)

                    # fitness of locations
                    fitness=self.objective_function(X[i,:])
                    self.conv.append(fitness)

                    # Update the location of Rabbit
                    if fitness>Rabbit_Energy: # Change this to > for maximization problem
                        Rabbit_Energy=fitness 
                        Rabbit_Location=X[i,:].copy() 

                E1=2*(1-(t/self.T)) # factor to show the decreaing energy of rabbit 
                #if (t % 50 == 0):
                        #print(['HHO En la iteración '+ str(t)+ ' la mejor aptitud es '+ str(Rabbit_Energy)])

                # Update the location of Harris' hawks 
                for i in range(0,self.N):

                    E0=2*rnd.random()-1  # -1<E0<1
                    Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper

                    # -------- Exploration phase Eq. (1) in paper -------------------

                    if abs(Escaping_Energy)>=1:
                        #Harris' hawks perch randomly based on 2 strategy:
                        q = rnd.random()
                        rand_Hawk_index = math.floor(self.N*rnd.random())
                        X_rand = X[rand_Hawk_index, :]
                        if q<0.5:
                            # perch based on other family members
                            X[i,:]=X_rand-rnd.random()*abs(X_rand-2*rnd.random()*X[i,:])

                        elif q>=0.5:
                            #perch on a random tall tree (random site inside group's home range)
                            X[i,:]=(Rabbit_Location - X.mean(0))-rnd.random()*((ub-lb)*rnd.random()+lb)

                    # -------- Exploitation phase -------------------
                    elif abs(Escaping_Energy)<1:
                        #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                        #phase 1: ----- surprise pounce (seven kills) ----------
                        #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                        r=rnd.random() # probablity of each event

                        if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                            X[i,:]=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X[i,:])

                        if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                            Jump_strength=2*(1- rnd.random()) # random jump strength of the rabbit
                            X[i,:]=(Rabbit_Location-X[i,:])-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])

                        #phase 2: --------performing team rapid dives (leapfrog movements)----------

                        if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                            #rabbit try to escape by many zigzag deceptive motions
                            Jump_strength=2*(1-rnd.random())
                            X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])
                            X1 = np.clip(X1, lb, ub)

                            if self.objective_function(X1)> fitness: # improved move?
                                X[i,:] = X1.copy()
                            else: # hawks perform levy-based short rapid dives around the rabbit
                                X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])+np.multiply(np.random.randn(dim),self.Levy(dim))
                                X2 = np.clip(X2, lb, ub)
                                if self.objective_function(X2)> fitness:
                                    X[i,:] = X2.copy()
                        if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                             Jump_strength=2*(1-rnd.random())
                             X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))
                             X1 = np.clip(X1, lb, ub)

                             if self.objective_function(X1)> fitness: # improved move?
                                X[i,:] = X1.copy()
                             else: # Perform levy-based short rapid dives around the rabbit
                                 X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))+np.multiply(np.random.randn(dim),self.Levy(dim))
                                 X2 = np.clip(X2, lb, ub)
                                 if self.objective_function(X2)> fitness:
                                    X[i,:] = X2.copy()

                convergence_curve[t]=Rabbit_Energy
                t=t+1


            return Rabbit_Energy, Rabbit_Location, convergence_curve


    @staticmethod
    def Levy(dim):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = 0.01 * np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        zz = np.power(np.absolute(v), (1 / beta))
        step = np.divide(u, zz)
        return step

import numpy as np

class CSA:
    def __init__(self, image, N, T, lb, ub, dim):
        self.image = image
        self.histogram = self.calculate_histogram(image)
        self.N = N
        self.T = T
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.AP = 0.1
        self.fl = 2

        self.x = np.random.uniform(lb, ub, (N, dim))
        self.mem = np.copy(self.x)
        self.fit_mem = [self.objective_function(xi) for xi in self.x]
        self.ffit = np.zeros(T)

    def calculate_histogram(self, image):
        return np.histogram(image, bins=256, range=(0, 256))[0]

    def objective_function(self, thresholds):
        # Histogram of the image
        histogram, _ = np.histogram(self.image, bins=256, range=(0, 256), density=True)
        thresholds = [int(round(t)) for t in thresholds]
        thresholds = sorted(thresholds)

        p_regions = []
        w_regions = []
        A_regions = []
        J = 0

        for region_idx in range(len(thresholds) + 1):
            start = 0 if region_idx == 0 else thresholds[region_idx - 1]
            end = thresholds[region_idx] if region_idx < len(thresholds) else 255

            p_region = [histogram[i] / np.sum(histogram) for i in range(start, end)]
            p_regions.append(p_region)
            w_regions.append(np.sum(p_region))

            A_region = [-(p / w_regions[region_idx]) * np.log(p / w_regions[region_idx]) if p > 0 else 0.001 for p in p_region]
            A_regions.append(A_region)

            J += np.sum(A_region)

        return J

    def evolve(self):
        #print('CSA está trabajando en tu problema')
        Conv_CSA = []

        for t in range(self.T):
            xnew = self.lb + (self.ub - self.lb) * np.random.rand(self.N, self.dim)
            xnew = np.clip(xnew, self.lb, self.ub)  # Ensure boundaries

            ft = [self.objective_function(xi) for xi in xnew]

            for i in range(self.N):
                if ft[i] > self.fit_mem[i]:
                    self.mem[i, :] = xnew[i, :]
                    self.fit_mem[i] = ft[i]

            best_fitness = max(self.fit_mem)
            self.ffit[t] = best_fitness
            Conv_CSA.append(best_fitness)

            #if t % 50 == 0:
                #print(f"En la iteración {t}, la mejor solución tiene una aptitud de {best_fitness}")

        ngbest = np.argmax(self.fit_mem)
        g_best = self.mem[ngbest, :]
        return best_fitness, g_best, Conv_CSA
def clean_and_show_binary_image(binary_image, min_size):

    if binary_image is None:
        print("Error: No se pudo cargar la imagen binaria.")
        return None

    # Remove small objects
    labels = skimage.morphology.label(binary_image)
    region_props = skimage.measure.regionprops(labels)
    for region in region_props:
        if region.area < min_size:
            for coord in region.coords:
                labels[coord[0], coord[1]] = 0

    # Convert to binary image
    cleaned_binary_image = np.where(labels >= 1, 1, 0)

    # Show the images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cleaned_binary_image, cmap='gray')
    plt.title('Imagen Limpia')
    plt.axis('off')

    plt.show()

    return cleaned_binary_image
def segment_image(image1, thresholds):
    # Crear una lista para almacenar las imágenes segmentadas
    segmented_images = []

    # Asignar etiquetas a cada región según los umbrales
    for i in range(len(thresholds) + 1):
        segmented_image = np.zeros_like(image1)  # Crear una nueva matriz en cada iteración

        if i == 0:
            segmented_image[image1 <= thresholds[i]] = 1
        elif i == len(thresholds):
            segmented_image[image1 > thresholds[i - 1]] = i + 1
        else:
            segmented_image[(image1 > thresholds[i - 1]) & (image1 <= thresholds[i])] = i + 1

        # Almacenar la imagen segmentada actual en la lista
        segmented_images.append(segmented_image)

    return segmented_images
def generate_single_channel_image(segmented_images):
    # Sumar todas las imágenes segmentadas para obtener una imagen en un solo canal
    result_image = np.sum(segmented_images, axis=0)

    # Escalar la imagen resultante para que esté en el rango [0, 255]
    result_image = (result_image / np.max(result_image) * 255).astype(np.uint8)
    result_image = np.asarray(result_image)
    return result_image
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
# Ruta a la carpeta que contiene las imágenes
carpeta_imagenes = 'img'
# Obtener la lista de archivos en la carpeta
archivos_todos = os.listdir(carpeta_imagenes)
archivos = [archivo for archivo in archivos_todos if archivo.endswith('.png')]
#parámetros de las metaheurísticas
N = 30
T = 100
dim = 7
lb = np.full(dim, 0)
ub = np.full(dim, 255)
ejec=30
#metricas
RSA_vec, HBA_vec, OPA_vec, BES_vec, GWO_vec, CSA_vec, HHO_vec, TSO_vec = [], [], [], [], [], [], [], []
RSA_umbral, HBA_umbral, OPA_umbral, BES_umbral, GWO_umbral, CSA_umbral, HHO_umbral, TSO_umbral = [], [], [], [], [], [], [], []
RSA_PSNR, HBA_PSNR, OPA_PSNR, BES_PSNR, GWO_PSNR, CSA_PSNR, HHO_PSNR, TSO_PSNR = [], [], [], [], [], [], [], []
RSA_SSIM, HBA_SSIM, OPA_SSIM, BES_SSIM, GWO_SSIM, CSA_SSIM, HHO_SSIM, TSO_SSIM = [], [], [], [], [], [], [], []
convergencia = []

for i in archivos:
    print('Imagen:',i)
    ruta_imagen = os.path.join(carpeta_imagenes, i)
    # Cargar la imagen en su formato original (incluyendo profundidad de bits)
    image = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    if image.shape[0] != 96 or image.shape[1] != 96:
        # Redimensionar la imagen a 96x96
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
        
    # Comprobar si la imagen es de 8 bits, si no, normalizar
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
 
       # Inicialización de listas para almacenar métricas de la imagen actual
    RSA_fitness, HBA_fitness, OPA_fitness, BES_fitness, GWO_fitness, CSA_fitness, HHO_fitness, TSO_fitness = [], [], [], [], [], [], [], []
    RSA_best, HBA_best, OPA_best, BES_best, GWO_best, CSA_best, HHO_best, TSO_best = [], [], [], [], [], [], [], []
    RSA_PSNR_img, HBA_PSNR_img, OPA_PSNR_img, BES_PSNR_img, GWO_PSNR_img, CSA_PSNR_img, HHO_PSNR_img, TSO_PSNR_img = [], [], [], [], [], [], [], []
    RSA_SSIM_img, HBA_SSIM_img, OPA_SSIM_img, BES_SSIM_img, GWO_SSIM_img, CSA_SSIM_img, HHO_SSIM_img, TSO_SSIM_img = [], [], [], [], [], [], [], []
    convergencia_img = []
    
    for j in range(0,ejec,1):
        print('ejecucion N°:',j)
# Inicializar y resolver utilizando la clase TSO
        tso = TSO(image, N, T, lb, ub, dim)
        Best_F_TSO, Best_P_TSO, Conv_TSO = tso.resolve()
        Best_P_TSO = sorted(Best_P_TSO)
        TSO_best.append(Best_P_TSO)
        segmented_images = segment_image(image, Best_P_TSO)
        image2=generate_single_channel_image(segmented_images)
        TSO_PSNR_img.append(calculate_psnr(image, image2))
        TSO_SSIM_img.append(calculate_ssim(image, image2))        
        TSO_fitness.append(Best_F_TSO)
        
# Inicializar y resolver utilizando la clase RSA
        rsa = RSA(image, N, T, lb, ub, dim)
        Best_F_RSA, Best_P_RSA, Conv_RSA = rsa.resolve()
        Best_P_RSA = sorted(Best_P_RSA)
        RSA_best.append(Best_P_RSA)
        segmented_images = segment_image(image, Best_P_RSA)
        image2=generate_single_channel_image(segmented_images)
        RSA_PSNR_img.append(calculate_psnr(image, image2))
        RSA_SSIM_img.append(calculate_ssim(image, image2))
        RSA_fitness.append(Best_F_RSA)
# Inicializar y resolver utilizando la clase OPA
        opa = OPA(image, N, T, lb, ub, dim)
        Best_F_OPA, Best_P_OPA, Conv_OPA = opa.resolve()
        Best_P_OPA = sorted(Best_P_OPA)
        OPA_best.append(Best_P_OPA)
        segmented_images = segment_image(image, Best_P_OPA)
        image2=generate_single_channel_image(segmented_images)
        OPA_PSNR_img.append(calculate_psnr(image, image2))
        OPA_SSIM_img.append(calculate_ssim(image, image2))        
        OPA_fitness.append(Best_F_OPA)
# Inicializar y resolver utilizando la clase HBA
        hba = HBA(image, N, T, lb, ub, dim)
        Best_F_HBA, Best_P_HBA, Conv_HBA = hba.resolve()
        Best_P_HBA = sorted(Best_P_HBA)
        HBA_best.append(Best_P_HBA)
        segmented_images = segment_image(image, Best_P_HBA)
        image2=generate_single_channel_image(segmented_images)
        HBA_PSNR_img.append(calculate_psnr(image, image2))
        HBA_SSIM_img.append(calculate_ssim(image, image2))        
        HBA_fitness.append(Best_F_HBA)
# Inicializar y resolver utilizando la clase BES
        bes= BES(image, N, T, lb, ub, dim)
        Best_F_BES, Best_P_BES, Conv_BES = bes.resolve()
        Best_P_BES = sorted(Best_P_BES)
        BES_best.append(Best_P_BES)
        segmented_images = segment_image(image, Best_P_BES)
        image2=generate_single_channel_image(segmented_images)
        BES_PSNR_img.append(calculate_psnr(image, image2))
        BES_SSIM_img.append(calculate_ssim(image, image2))        
        BES_fitness.append(Best_F_BES)
# Inicializar y resolver utilizando la clase GrayWolfOptimizer
        gwo = GWO(image, N, T, lb, ub, dim)
        Best_F_GWO, Best_P_GWO, Conv_GWO = gwo.solve()
        Best_P_GWO = sorted(Best_P_GWO)
        GWO_best.append(Best_P_GWO)
        segmented_images = segment_image(image, Best_P_GWO)
        image2=generate_single_channel_image(segmented_images)
        GWO_PSNR_img.append(calculate_psnr(image, image2))
        GWO_SSIM_img.append(calculate_ssim(image, image2))        
        GWO_fitness.append(Best_F_GWO)
# Inicializar y resolver utilizando la clase CSA
        csa = CSA(image, N, T, lb, ub, dim)
        Best_F_CSA, Best_P_CSA, Conv_CSA = csa.evolve()
        Best_P_CSA = sorted(Best_P_CSA)
        CSA_best.append(Best_P_CSA)
        segmented_images = segment_image(image, Best_P_CSA)
        image2=generate_single_channel_image(segmented_images)
        CSA_PSNR_img.append(calculate_psnr(image, image2))
        CSA_SSIM_img.append(calculate_ssim(image, image2))        
        CSA_fitness.append(Best_F_CSA)
# Inicializar y resolver utilizando la clase HHO
        hho = HHO(image, N, T, lb, ub, dim)
        Best_F_HHO, Best_P_HHO, Conv_HHO = hho.optimize()
        Best_P_HHO = sorted(Best_P_HHO)
        HHO_best.append(Best_P_HHO)
        segmented_images = segment_image(image, Best_P_HHO)
        image2=generate_single_channel_image(segmented_images)
        HHO_PSNR_img.append(calculate_psnr(image, image2))
        HHO_SSIM_img.append(calculate_ssim(image, image2))        
        HHO_fitness.append(Best_F_HHO)
        lista1=[Conv_RSA,Conv_HBA,Conv_OPA,Conv_CSA,Conv_HHO,Conv_TSO,Conv_GWO,Conv_BES]
        convergencia_img.append(lista1)
    print('Imagen ',i, 'ejecución ',j)
 
    RSA_vec.append(RSA_fitness)
    HBA_vec.append(HBA_fitness)
    OPA_vec.append(OPA_fitness)
    BES_vec.append(BES_fitness)
    GWO_vec.append(GWO_fitness)
    CSA_vec.append(CSA_fitness)
    HHO_vec.append(HHO_fitness)
    TSO_vec.append(TSO_fitness)

    RSA_PSNR.append(RSA_PSNR_img)
    HBA_PSNR.append(HBA_PSNR_img)
    OPA_PSNR.append(OPA_PSNR_img)
    BES_PSNR.append(BES_PSNR_img)
    GWO_PSNR.append(GWO_PSNR_img)
    CSA_PSNR.append(CSA_PSNR_img)
    HHO_PSNR.append(HHO_PSNR_img)
    TSO_PSNR.append(TSO_PSNR_img)

    RSA_SSIM.append(RSA_SSIM_img)
    HBA_SSIM.append(HBA_SSIM_img)
    OPA_SSIM.append(OPA_SSIM_img)
    BES_SSIM.append(BES_SSIM_img)
    GWO_SSIM.append(GWO_SSIM_img)
    CSA_SSIM.append(CSA_SSIM_img)
    HHO_SSIM.append(HHO_SSIM_img)
    TSO_SSIM.append(TSO_SSIM_img)
    
    RSA_umbral.append(RSA_best)
    HBA_umbral.append(HBA_best)
    OPA_umbral.append(OPA_best)
    BES_umbral.append(BES_best)
    GWO_umbral.append(GWO_best)
    CSA_umbral.append(CSA_best)
    HHO_umbral.append(HHO_best)
    TSO_umbral.append(TSO_best)
    

    convergencia.append(convergencia_img)

# Estructura del diccionario para el DataFrame
data = {
    'Imagen': [],
    'Ejecucion': [],
    'RSA_fitness': [],
    'HBA_fitness': [],
    'OPA_fitness': [],
    'BES_fitness': [],
    'GWO_fitness': [],
    'CSA_fitness': [],
    'HHO_fitness': [],
    'TSO_fitness': [],
    'RSA_PSNR': [],
    'HBA_PSNR': [],
    'OPA_PSNR': [],
    'BES_PSNR': [],
    'GWO_PSNR': [],
    'CSA_PSNR': [],
    'HHO_PSNR': [],
    'TSO_PSNR': [],
    'RSA_SSIM': [],
    'HBA_SSIM': [],
    'OPA_SSIM': [],
    'BES_SSIM': [],
    'GWO_SSIM': [],
    'CSA_SSIM': [],
    'HHO_SSIM': [],
    'TSO_SSIM': [],
    'RSA_umbral': [],
    'HBA_umbral': [],
    'OPA_umbral': [],
    'BES_umbral': [],
    'GWO_umbral': [],
    'CSA_umbral': [],
    'HHO_umbral': [],
    'TSO_umbral': [],
    'Convergencia': []
}

# Llenar el diccionario con los datos
for idx, img in enumerate(archivos):
    for ejecucion in range(ejec):
        data['Imagen'].append(img)
        data['Ejecucion'].append(ejecucion)
        data['RSA_fitness'].append(RSA_vec[idx][ejecucion] if idx < len(RSA_vec) and ejecucion < len(RSA_vec[idx]) else None)
        data['HBA_fitness'].append(HBA_vec[idx][ejecucion] if idx < len(HBA_vec) and ejecucion < len(HBA_vec[idx]) else None)
        data['OPA_fitness'].append(OPA_vec[idx][ejecucion] if idx < len(OPA_vec) and ejecucion < len(OPA_vec[idx]) else None)
        data['BES_fitness'].append(BES_vec[idx][ejecucion] if idx < len(BES_vec) and ejecucion < len(BES_vec[idx]) else None)
        data['GWO_fitness'].append(GWO_vec[idx][ejecucion] if idx < len(GWO_vec) and ejecucion < len(GWO_vec[idx]) else None)
        data['CSA_fitness'].append(CSA_vec[idx][ejecucion] if idx < len(CSA_vec) and ejecucion < len(CSA_vec[idx]) else None)
        data['HHO_fitness'].append(HHO_vec[idx][ejecucion] if idx < len(HHO_vec) and ejecucion < len(HHO_vec[idx]) else None)
        data['TSO_fitness'].append(TSO_vec[idx][ejecucion] if idx < len(TSO_vec) and ejecucion < len(TSO_vec[idx]) else None)

        data['RSA_PSNR'].append(RSA_PSNR[idx][ejecucion] if idx < len(RSA_PSNR) and ejecucion < len(RSA_PSNR[idx]) else None)
        data['HBA_PSNR'].append(HBA_PSNR[idx][ejecucion] if idx < len(HBA_PSNR) and ejecucion < len(HBA_PSNR[idx]) else None)
        data['OPA_PSNR'].append(OPA_PSNR[idx][ejecucion] if idx < len(OPA_PSNR) and ejecucion < len(OPA_PSNR[idx]) else None)
        data['BES_PSNR'].append(BES_PSNR[idx][ejecucion] if idx < len(BES_PSNR) and ejecucion < len(BES_PSNR[idx]) else None)
        data['GWO_PSNR'].append(GWO_PSNR[idx][ejecucion] if idx < len(GWO_PSNR) and ejecucion < len(GWO_PSNR[idx]) else None)
        data['CSA_PSNR'].append(CSA_PSNR[idx][ejecucion] if idx < len(CSA_PSNR) and ejecucion < len(CSA_PSNR[idx]) else None)
        data['HHO_PSNR'].append(HHO_PSNR[idx][ejecucion] if idx < len(HHO_PSNR) and ejecucion < len(HHO_PSNR[idx]) else None)
        data['TSO_PSNR'].append(TSO_PSNR[idx][ejecucion] if idx < len(TSO_PSNR) and ejecucion < len(TSO_PSNR[idx]) else None)

        data['RSA_SSIM'].append(RSA_SSIM[idx][ejecucion] if idx < len(RSA_SSIM) and ejecucion < len(RSA_SSIM[idx]) else None)
        data['HBA_SSIM'].append(HBA_SSIM[idx][ejecucion] if idx < len(HBA_SSIM) and ejecucion < len(HBA_SSIM[idx]) else None)
        data['OPA_SSIM'].append(OPA_SSIM[idx][ejecucion] if idx < len(OPA_SSIM) and ejecucion < len(OPA_SSIM[idx]) else None)
        data['BES_SSIM'].append(BES_SSIM[idx][ejecucion] if idx < len(BES_SSIM) and ejecucion < len(BES_SSIM[idx]) else None)
        data['GWO_SSIM'].append(GWO_SSIM[idx][ejecucion] if idx < len(GWO_SSIM) and ejecucion < len(GWO_SSIM[idx]) else None)
        data['CSA_SSIM'].append(CSA_SSIM[idx][ejecucion] if idx < len(CSA_SSIM) and ejecucion < len(CSA_SSIM[idx]) else None)
        data['HHO_SSIM'].append(HHO_SSIM[idx][ejecucion] if idx < len(HHO_SSIM) and ejecucion < len(HHO_SSIM[idx]) else None)
        data['TSO_SSIM'].append(TSO_SSIM[idx][ejecucion] if idx < len(TSO_SSIM) and ejecucion < len(TSO_SSIM[idx]) else None)

        data['RSA_umbral'].append(RSA_umbral[idx][ejecucion] if idx < len(RSA_umbral) and ejecucion < len(RSA_umbral[idx]) else None)
        data['HBA_umbral'].append(HBA_umbral[idx][ejecucion] if idx < len(HBA_umbral) and ejecucion < len(HBA_umbral[idx]) else None)
        data['OPA_umbral'].append(OPA_umbral[idx][ejecucion] if idx < len(OPA_umbral) and ejecucion < len(OPA_umbral[idx]) else None)
        data['BES_umbral'].append(BES_umbral[idx][ejecucion] if idx < len(BES_umbral) and ejecucion < len(BES_umbral[idx]) else None)
        data['GWO_umbral'].append(GWO_umbral[idx][ejecucion] if idx < len(GWO_umbral) and ejecucion < len(GWO_umbral[idx]) else None)
        data['CSA_umbral'].append(CSA_umbral[idx][ejecucion] if idx < len(CSA_umbral) and ejecucion < len(CSA_umbral[idx]) else None)
        data['HHO_umbral'].append(HHO_umbral[idx][ejecucion] if idx < len(HHO_umbral) and ejecucion < len(HHO_umbral[idx]) else None)
        data['TSO_umbral'].append(TSO_umbral[idx][ejecucion] if idx < len(TSO_umbral) and ejecucion < len(TSO_umbral[idx]) else None)

        data['Convergencia'].append(convergencia[idx][ejecucion] if idx < len(convergencia) and ejecucion < len(convergencia[idx]) else None)

# Creamos el DataFrame de pandas
df = pd.DataFrame(data)

# Muestra las primeras filas del DataFrame para verificar
df.head()
carpeta = 'Metrica_8_bits_7_dim_kapur'

# Guarda el DataFrame en un archivo CSV
try:
    # Intenta guardar el DataFrame como un archivo Excel
    df.to_excel(carpeta + '.xlsx', index=False)
    print("Los datos se han guardado en un archivo Excel.")
except Exception as e:
    print(f"Hubo un problema al guardar en Excel: {e}")
    print("Se intentará guardar en formato CSV.")
    # Si falla, intenta guardar como CSV
    df.to_csv(carpeta + '.csv', index=False)
    print("Los datos se han guardado en un archivo CSV.")

algorithms = ['RSA', 'HBA', 'OPA', 'CSA', 'HHO', 'TSO', 'GWO', 'BES']

for idx, convergencias_por_imagen in enumerate(convergencia):
    print('Procesando imagen', idx)

    for i, lista1 in enumerate(convergencias_por_imagen):
        lista2 = []

        for arr in lista1:
            arr = np.asarray(arr)
            lista2.append(arr)

        # Graficar cada numpy array en la lista
        plt.figure(figsize=(10, 6))  # Ajusta el tamaño del gráfico
        for j, arr in enumerate(lista2):
            x = np.arange(arr.shape[0])  # Los valores en el eje x serán 0, 1, 2, ...
            y = arr  # Los valores en el eje y serán los valores del numpy array

            plt.plot(x, y, label=algorithms[j])

        # Configurar etiquetas y leyenda
        plt.xlabel('Iteraciones')
        plt.ylabel('Valor función objetivo')
        plt.title('Convergencia para la imagen ' + str(idx) + ', ejecución ' + str(i))
        plt.legend()
        plt.grid(True)

        # Mostrar el gráfico
        plt.show()
import numpy as np
import os

# Create the folder "Metrica_8_bits_7_dim_otsu" if it doesn't exist
os.makedirs(carpeta, exist_ok=True)

# Save the arrays in the "Metrica_8_bits_7_dim_otsu" folder
np.save(os.path.join(carpeta, 'RSA_vec.npy'), RSA_vec)
np.save(os.path.join(carpeta, 'HBA_vec.npy'), HBA_vec)
np.save(os.path.join(carpeta, 'OPA_vec.npy'), OPA_vec)
np.save(os.path.join(carpeta, 'BES_vec.npy'), BES_vec)
np.save(os.path.join(carpeta, 'GWO_vec.npy'), GWO_vec)
np.save(os.path.join(carpeta, 'CSA_vec.npy'), CSA_vec)
np.save(os.path.join(carpeta, 'HHO_vec.npy'), HHO_vec)
np.save(os.path.join(carpeta, 'TSO_vec.npy'), TSO_vec)

np.save(os.path.join(carpeta, 'RSA_PSNR.npy'), RSA_PSNR)
np.save(os.path.join(carpeta, 'HBA_PSNR.npy'), HBA_PSNR)
np.save(os.path.join(carpeta, 'OPA_PSNR.npy'), OPA_PSNR)
np.save(os.path.join(carpeta, 'BES_PSNR.npy'), BES_PSNR)
np.save(os.path.join(carpeta, 'GWO_PSNR.npy'), GWO_PSNR)
np.save(os.path.join(carpeta, 'CSA_PSNR.npy'), CSA_PSNR)
np.save(os.path.join(carpeta, 'HHO_PSNR.npy'), HHO_PSNR)
np.save(os.path.join(carpeta, 'TSO_PSNR.npy'), TSO_PSNR)

np.save(os.path.join(carpeta, 'RSA_SSIM.npy'), RSA_SSIM)
np.save(os.path.join(carpeta, 'HBA_SSIM.npy'), HBA_SSIM)
np.save(os.path.join(carpeta, 'OPA_SSIM.npy'), OPA_SSIM)
np.save(os.path.join(carpeta, 'BES_SSIM.npy'), BES_SSIM)
np.save(os.path.join(carpeta, 'GWO_SSIM.npy'), GWO_SSIM)
np.save(os.path.join(carpeta, 'CSA_SSIM.npy'), CSA_SSIM)
np.save(os.path.join(carpeta, 'HHO_SSIM.npy'), HHO_SSIM)
np.save(os.path.join(carpeta, 'TSO_SSIM.npy'), TSO_SSIM)

np.save(os.path.join(carpeta, 'convergencia.npy'), convergencia)
