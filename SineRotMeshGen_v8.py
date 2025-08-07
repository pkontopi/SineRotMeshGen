# Author: Patrick Kontopidis © 2025

import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import random
#import meshio
import gmsh
import sys
import tkinter as tk
import sounddevice as sd
import threading
import time 
import pyautogui
from scipy.signal import find_peaks
import os 
import concurrent.futures
import cv2
import shutil
import glob
import kaleido
import plotly.io as pio

class mesh:
    def __init__(self): 
        pass

    def plotten3d(self, x_geo, y_geo, z_geo):
        index_arr = range(len(x_geo))
        fig = px.scatter_3d(x=x_geo,y=y_geo,z=z_geo, hover_data=[index_arr])
        fig.update_traces(marker=dict(size=2))
        fig.show()

    def plotten3d_multi(self, x_geo, y_geo, z_geo, x_btm, y_btm, z_btm):
        #index_arr = range(len(x_1))
        fig = go.Figure()

        trace = go.Scatter3d(x=x_geo,y=y_geo,z=z_geo, mode='markers')

        if isinstance(x_btm[0], np.ndarray): 
            for i in range(len(x_btm)): 
                trace_2 = go.Scatter3d(x=x_btm[i], y=y_btm[i], z=z_btm[i], mode='markers')
                fig.add_trace(trace_2)
        else: 
            trace_2 = go.Scatter3d(x=x_btm,y=y_btm,z=z_btm)
            fig.add_trace(trace_2)

        fig.add_trace(trace)
        fig.update_traces(marker=dict(size=2))
        fig.update_layout(
                    title_font_size=20, 
                    title_x=0.5,
                        scene=dict(
                            xaxis_title="X [m]",
                            yaxis_title="Y [m]",
                            zaxis_title="Z [m]",
                        ),
                    margin=dict(l=0, r=0, t=50, b=0), 
                    showlegend=False,
                    scene_camera=dict(
                        eye=dict(x=1.6, y=1.6, z=1.6)  # Kamera weiter zurücksetzen für weniger Zoom
                    ))
        fig.show()

    def plotten2d(self, x_data, data):
        fig = go.Figure()
        trace = go.Scatter(x=x_data, y=data)
        fig.add_trace(trace)
        fig.show()

    def _surface_nodes(self, n, k, durchmesser, **kwargs):
        pi = np.pi
        x = np.array([])
        y = np.array([])
        z = np.array([])

        phi_arr = []
        crc = []

        #zeiger_theta = np.linspace(0, periode*2*np.pi, k)
        #theta = hoehe*np.cos(zeiger_theta)
        hoehe = kwargs.get('hoehe', 50/2*0.001)
        periode = kwargs.get('periode', 0)
        # if periode > 0: 
            
        #     bottom_position = 1
        # else: 
        #     bottom_position = kwargs.get('bottom_position', -1.2)

        bottom_position = kwargs.get('bottom_position', -1.2)

        th = kwargs.get('theta', np.cos(np.linspace(0, periode*2*pi, k)))
        theta = hoehe*th

        zeiger = np.linspace(0, durchmesser, k)

        for i in range(k):
            # Abstand der Punkte auf dem Kreis zueinander proportional zum Radius, damit Abstand immer gleich ist
            if(i==0):
                phi = np.array([0.])
                phi_arr.append(phi)
                z = np.concatenate((z, theta[i]*np.ones(1)+np.array([bottom_position])))

                crc.append(1)
            elif(i == 1):
                phi = np.linspace(0, 2*pi-(2*pi/n), n)
                phi_arr.append(phi)
                z = np.concatenate((z, theta[i]*np.ones(n)+np.array([bottom_position]*(n))))
                crc.append(n)
            else:
                phi = np.linspace(0, 2*pi - (2*pi/(n*i)), n*i)
                phi_arr.append(phi)
                z = np.concatenate((z, theta[i]*np.ones(n*i)+np.array([bottom_position]*(n*i))))
                crc.append(n*i)
            x = np.concatenate((x, zeiger[i]*np.sin(phi)))
            y = np.concatenate((y, zeiger[i]*np.cos(phi)))

        return x, y, z, phi_arr, crc

    def _surface_nodes_2d(self, n, durchmesser, **kwargs): 
        pi = np.pi

        x = np.array([])
        y = np.array([])
        z = np.array([])

        phi_arr = []
        crc = []
        k = (n*4) #*periode  # Anzahl Ringe

        hoehe = kwargs.get('hoehe', 50/2*0.001)
        periode = kwargs.get('periode', 0)
        zeiger = np.linspace(0, durchmesser, k)

        # Berge und Täler
        for i in range(k):

            if periode > 1 and i < k/periode: 
                zeiger_theta = np.linspace(0, (periode/2)*2*pi, k)
                theta = hoehe*np.cos(zeiger_theta)
            else: 
                zeiger_theta = np.linspace(0, periode*2*pi, k)
                theta = hoehe*np.cos(zeiger_theta)
            
            # Abstand der Punkte auf dem Kreis zueinander proportional zum Radius, damit Abstand immer gleich ist
            if(i==0):
                phi = np.array([0.])
                phi_arr.append(phi)
                # Berg + Tal
                if periode > 1: 
                    z = np.concatenate((z, theta[i]*np.ones(1)))
                else: 
                    alpha = np.linspace(0, periode*2*pi-(periode*2*pi/(1)), 1)
                    z = np.concatenate((z, theta[i]*np.sin(alpha)))
                crc.append(1)
            elif(i == 1):
                phi = np.linspace(0, 2*pi-(2*pi/n), n)
                phi_arr.append(phi)
                # Berg + Tal

                if periode > 1: 
                    z = np.concatenate((z, theta[i]*np.ones(n)))
                else: 
                    alpha = np.linspace(0, periode*2*pi-(periode*2*pi/(n)), n)
                    z = np.concatenate((z, theta[i]*np.sin(alpha)))
                crc.append(n)
            elif(periode > 1 and i < k/periode):
                phi = np.linspace(0, 2*pi - (2*pi/(n*i)), n*i)
                phi_arr.append(phi)
                # Berg + Tal
                z = np.concatenate((z, theta[i]*np.ones(n*i)))
                crc.append(n*i)
            else:
                phi = np.linspace(0, 2*pi - (2*pi/(n*i)), n*i)
                phi_arr.append(phi)
                # Berg + Tal

                zeiger_theta = np.linspace(0, periode*2*pi, k)
                theta = hoehe*np.cos(zeiger_theta) 

                alpha = np.linspace(0, periode*2*pi-(periode*2*pi/(n*i)), n*i)
                z = np.concatenate((z, theta[i]*np.sin(alpha)))

                crc.append(n*i)
            #Berg + Tal
            x = np.concatenate((x, zeiger[i]*np.sin(phi)))
            y = np.concatenate((y, zeiger[i]*np.cos(phi)))   

        return x, y, z, phi_arr, crc

    def _surface_nodes_2d_rec(self, n, durchmesser, **kwargs): 
        pi = np.pi

        x = np.array([])
        y = np.array([])
        z = np.array([])

        phi_arr = []
        crc = []
        k = (n*4) #*periode  # Anzahl Ringe

        hoehe = kwargs.get('hoehe', 50/2*0.001)
        periode = kwargs.get('periode', 0)
        zeiger = np.linspace(0, durchmesser, k)

        # Berge und Täler
        for i in range(k):
            if periode > 1 and i < k/periode: 
                zeiger_theta = np.linspace(0, (periode/2)*2*pi, k)
                th = kwargs.get('theta', np.cos(zeiger_theta))
                #theta = hoehe*np.cos(zeiger_theta)
                theta = hoehe*th
            else: 
                zeiger_theta = np.linspace(0, periode*2*pi, k)
                #theta = hoehe*np.cos(zeiger_theta)
            
            # Abstand der Punkte auf dem Kreis zueinander proportional zum Radius, damit Abstand immer gleich ist
            if(i==0):
                phi = np.array([0.])
                phi_arr.append(phi)

                if periode > 1: 
                    z = np.concatenate((z, theta[i]*np.ones(1)))
                else: 
                    alpha = np.linspace(0, periode*2*pi-(periode*2*pi/(1)), 1)
                    z = np.concatenate((z, theta[i]*np.sin(alpha)))
                crc.append(1)
            elif(i == 1):
                phi = np.linspace(0, 2*pi-(2*pi/n), n)
                phi_arr.append(phi)

                if periode > 1: 
                    z = np.concatenate((z, theta[i]*np.ones(n)))
                else: 
                    alpha = np.linspace(0, periode*2*pi-(periode*2*pi/(n)), n)
                    z = np.concatenate((z, theta[i]*np.sin(alpha)))
                crc.append(n)
            elif(periode > 1 and i < k/periode):
                phi = np.linspace(0, 2*pi - (2*pi/(n*i)), n*i)
                phi_arr.append(phi)

                z = np.concatenate((z, theta[i]*np.ones(n*i)))
                crc.append(n*i)
            else:
                phi = np.linspace(0, 2*pi - (2*pi/(n*i)), n*i)
                phi_arr.append(phi)

                zeiger_theta = np.linspace(0, periode*2*pi, k)
                #theta = hoehe*np.cos(zeiger_theta) 

                alpha = np.linspace(0, periode*2*pi-(periode*2*pi/(n*i)), n*i)

                # print("str(len(alpha))"+str(len(alpha)))
                # print("str(len(theta))"+str(len(theta)))
                z = np.concatenate((z, theta[i]*np.sin(alpha)))
                
                #print("theta i: "+str(theta[i]))

                crc.append(n*i)

            x = np.concatenate((x, zeiger[i]*np.sin(phi)))
            y = np.concatenate((y, zeiger[i]*np.cos(phi)))   

        return x, y, z, phi_arr, crc

    def _surface_nodes_2d_bottom(self, n, durchmesser, **kwargs): 
        pi = np.pi

        x = np.array([])
        y = np.array([])
        z = np.array([])

        phi_arr = []
        crc = []
        k = (n*4) #*periode  # Anzahl Ringe

        hoehe = kwargs.get('hoehe', 50/2*0.001)
        periode = kwargs.get('periode', 0)
        zeiger = np.linspace(0, durchmesser, k)

        bottom_position = kwargs.get('bottom_position', -1.2)

        # Berge und Täler
        for i in range(k):

            if periode > 1 and i < k/periode: 
                zeiger_theta = np.linspace(0, (periode/2)*2*pi, k)
                theta = hoehe*np.cos(zeiger_theta)
            else: 
                zeiger_theta = np.linspace(0, periode*2*pi, k)
                theta = hoehe*np.cos(zeiger_theta)
            
            # Abstand der Punkte auf dem Kreis zueinander proportional zum Radius, damit Abstand immer gleich ist
            if(i==0):
                phi = np.array([0.])
                phi_arr.append(phi)
                # Berg + Tal
                if periode > 1: 
                    z = np.concatenate((z, theta[i]*np.ones(1)+np.array([bottom_position])))
                else: 
                    alpha = np.linspace(0, periode*2*pi-(periode*2*pi/(1)), 1)
                    z = np.concatenate((z, theta[i]*np.sin(alpha)+np.array([bottom_position])))
                crc.append(1)
            elif(i == 1):
                phi = np.linspace(0, 2*pi-(2*pi/n), n)
                phi_arr.append(phi)
                # Berg + Tal

                if periode > 1: 
                    z = np.concatenate((z, theta[i]*np.ones(n)+np.array([bottom_position]*(n))))
                else: 
                    alpha = np.linspace(0, periode*2*pi-(periode*2*pi/(n)), n)
                    z = np.concatenate((z, theta[i]*np.sin(alpha)+np.array([bottom_position]*(n))))
                crc.append(n)
            elif(periode > 1 and i < k/periode):
                phi = np.linspace(0, 2*pi - (2*pi/(n*i)), n*i)
                phi_arr.append(phi)
                # Berg + Tal
                z = np.concatenate((z, theta[i]*np.ones(n*i)+np.array([bottom_position]*(n*i))))
                crc.append(n*i)
            else:
                phi = np.linspace(0, 2*pi - (2*pi/(n*i)), n*i)
                phi_arr.append(phi)
                # Berg + Tal

                zeiger_theta = np.linspace(0, periode*2*pi, k)
                theta = hoehe*np.cos(zeiger_theta) 

                alpha = np.linspace(0, periode*2*pi-(periode*2*pi/(n*i)), n*i)
                z = np.concatenate((z, theta[i]*np.sin(alpha)+np.array([bottom_position]*(n*i))))

                crc.append(n*i)
            #Berg + Tal
            x = np.concatenate((x, zeiger[i]*np.sin(phi)))
            y = np.concatenate((y, zeiger[i]*np.cos(phi)))   

        return x, y, z, phi_arr, crc

    def nodes(self, **kwargs): 
        prusa_res = kwargs.get('prusa_res', False)
        plotten_ja_nein = kwargs.get('plot', False)
        periode = kwargs.get('period', 1)
        res = kwargs.get('resolution', 10)
        bottom_position = kwargs.get('bottom_position', -1.2) # positiver Wert für Metamaterial, negativer Wert für Komplementärmesh
        bottoms = kwargs.get('bottoms', 1)
        space = kwargs.get('space', 0.2)
        d = kwargs.get('durchmesser', 102)
        h = kwargs.get('hoehe', 50)

        k = res*4
        zeiger_theta = np.linspace(0, periode*2*np.pi, k)
        theta = kwargs.get('theta', np.cos(zeiger_theta))
        k = len(theta)

        # Auflösung Prusa Slicer in mm, *0.001 für m
        if prusa_res: 
            durchmesser = d/2
            hoehe = h/2
        else: 
            durchmesser = d/2*0.001
            hoehe = h/2*0.001

        # Berge + Täler
        [x_geo, y_geo, z_geo, phi_arr_geo, crc_geo] = self._surface_nodes(res, k, durchmesser, periode=periode, hoehe=hoehe, theta=theta)

        # Boden / Böden: 
        if bottoms > 1: 
            x_btm=[None]*bottoms
            y_btm=[None]*bottoms
            z_btm=[None]*bottoms

            sub = hoehe/bottoms
            #sub_d = durchmesser*20/bottoms
            #durchmesser = durchmesser*20

            if bottom_position < 0: 
                space = -space
            for i in range(bottoms): 
                hoehe = hoehe-sub
                [x_btm[i], y_btm[i], z_btm[i], phi_arr_btm, crc_btm] = self._surface_nodes(res,k, durchmesser, periode=periode, hoehe=hoehe, bottom_position=bottom_position-(space*i))
        else: 
            [x_btm, y_btm, z_btm, phi_arr_btm, crc_btm] = self._surface_nodes(res,k, durchmesser, periode=0, hoehe=hoehe, bottom_position=bottom_position)

        if plotten_ja_nein == True: 
            self.plotten3d_multi(x_geo, y_geo, z_geo, x_btm, y_btm,z_btm)

        return x_geo, y_geo, z_geo, crc_geo, phi_arr_geo, x_btm, y_btm, z_btm, crc_btm, phi_arr_btm

    def nodes_2D(self, **kwargs):
        prusa_res = kwargs.get('prusa_res', False)
        plotten_ja_nein = kwargs.get('plot', False)
        periode = kwargs.get('period', 1)
        res = kwargs.get('resolution', 10)
        bottom_position = kwargs.get('bottom_position', -1.2) # positiver Wert für Metamaterial, negativer Wert für Komplementärmesh
        bottoms = kwargs.get('bottoms', 1)
        space = kwargs.get('space', 0.2)
        d = kwargs.get('durchmesser', 102)
        h = kwargs.get('hoehe', 50)

        print("period: "+str(periode))

        k = res*4
        zeiger_theta = np.linspace(0, periode*2*np.pi, k)
        theta = kwargs.get('theta', np.cos(zeiger_theta))
        #k = int(len(theta))

        # Auflösung Prusa Slicer in mm, *0.001 für m
        if prusa_res: 
            durchmesser = d*2 #102/2*0.001
            hoehe = h/2 #*0.01
        else: 
            durchmesser = d/2*0.001
            hoehe = h/2*0.001

        [x_geo, y_geo, z_geo, phi_arr_geo, crc_geo] = self._surface_nodes_2d(res, durchmesser, periode=periode, hoehe=hoehe, theta=theta)
        # Boden
        #[x_btm, y_btm, z_btm, phi_arr_btm, crc_btm] = self._surface_nodes(res, k, durchmesser, periode=0, hoehe=hoehe, bottom_position=bottom_position)

        # Boden / Böden: 
        if bottoms > 1: 
            x_btm=[None]*bottoms
            y_btm=[None]*bottoms
            z_btm=[None]*bottoms

            sub = hoehe/bottoms

            if bottom_position < 0: 
                space = -space
            for i in range(bottoms): 
                hoehe = hoehe-sub
                #[x_btm[i], y_btm[i], z_btm[i], phi_arr_btm, crc_btm] = self._surface_nodes(res,k, durchmesser, periode=0, hoehe=hoehe, bottom_position=bottom_position+(space*i))
                [x_btm[i], y_btm[i], z_btm[i], phi_arr_btm, crc_btm] = self._surface_nodes_2d_bottom(res, durchmesser, periode=periode, hoehe=hoehe, bottom_position=bottom_position+(space*i))
        else: 
            [x_btm, y_btm, z_btm, phi_arr_btm, crc_btm] = self._surface_nodes(res,k, durchmesser, periode=0, hoehe=hoehe, bottom_position=bottom_position)

        if plotten_ja_nein == True: 
            self.plotten3d_multi(x_geo, y_geo, z_geo, x_btm, y_btm, z_btm)

        return x_geo, y_geo, z_geo, crc_geo, phi_arr_geo, x_btm, y_btm, z_btm, phi_arr_btm, crc_btm, hoehe
    
    def nodes_2D_rec(self, **kwargs):
        prusa_res = kwargs.get('prusa_res', False)
        plotten_ja_nein = kwargs.get('plot', False)
        periode = kwargs.get('period', 1)
        res = kwargs.get('resolution', 10)
        bottom_position = kwargs.get('bottom_position', -1.2) # positiver Wert für Metamaterial, negativer Wert für Komplementärmesh
        bottoms = kwargs.get('bottoms', 1)
        space = kwargs.get('space', 0.2)
        d = kwargs.get('durchmesser', 102)
        h = kwargs.get('hoehe', 50)

        print("period: "+str(periode))

        k = res*4
        zeiger_theta = np.linspace(0, periode*2*np.pi, k)
        theta = kwargs.get('theta', np.cos(zeiger_theta))
        #k = int(len(theta))

        # Auflösung Prusa Slicer in mm, *0.001 für m
        if prusa_res: 
            durchmesser = d*2 #102/2*0.001
            hoehe = h/2 #*0.01
        else: 
            durchmesser = d/2*0.001
            hoehe = h/2*0.001

        [x_geo, y_geo, z_geo, phi_arr_geo, crc_geo] = self._surface_nodes_2d_rec(res, durchmesser, periode=periode, hoehe=hoehe, theta=theta)
        # Boden
        #[x_btm, y_btm, z_btm, phi_arr_btm, crc_btm] = self._surface_nodes(res, k, durchmesser, periode=0, hoehe=hoehe, bottom_position=bottom_position)

        # Boden / Böden: 
        if bottoms > 1: 
            x_btm=[None]*bottoms
            y_btm=[None]*bottoms
            z_btm=[None]*bottoms

            sub = hoehe/bottoms

            if bottom_position < 0: 
                space = -space
            for i in range(bottoms): 
                hoehe = hoehe-sub
                #[x_btm[i], y_btm[i], z_btm[i], phi_arr_btm, crc_btm] = self._surface_nodes(res,k, durchmesser, periode=0, hoehe=hoehe, bottom_position=bottom_position+(space*i))
                [x_btm[i], y_btm[i], z_btm[i], phi_arr_btm, crc_btm] = self._surface_nodes_2d_bottom(res, durchmesser, periode=periode, hoehe=hoehe, bottom_position=bottom_position+(space*i))
        else: 
            [x_btm, y_btm, z_btm, phi_arr_btm, crc_btm] = self._surface_nodes(res,k, durchmesser, periode=0, hoehe=hoehe, bottom_position=bottom_position)

        if plotten_ja_nein == True: 
            self.plotten3d_multi(x_geo, y_geo, z_geo, x_btm, y_btm, z_btm)

        return x_geo, y_geo, z_geo, crc_geo, phi_arr_geo, x_btm, y_btm, z_btm, phi_arr_btm, crc_btm, hoehe
    # funzt    
    def material_old(self, x_geo, y_geo, z_geo, crc_geo, phi_arr_geo, x_btm, y_btm, z_btm, crc_btm, phi_arr_btm, **kwargs): 
        # GMSH Generation von meinem Meshcode! Hurra! Es läuft! 
        if len(crc_btm) != len(crc_geo) or len(phi_arr_geo) != len(phi_arr_btm): 
            return "Error: crc and phi_arr don't have same length."

        name = kwargs.get('saveas', 'default')
        gmsh_gui = kwargs.get('gmsh_gui', True)
        periode = kwargs.get('period', 1)
        resolution = kwargs.get('resolution', 10)
        komp = kwargs.get('counterpart', False)
        # twodom = kwargs.get('two_domains', False)
        rec = kwargs.get('rec', False)
        order = kwargs.get('mesh_order', 3)
        stl = kwargs.get('generate_stl', False)

        if komp: 
            kvar = -1
        else:
            kvar = 1

        if rec: 
            resolution = len(crc_geo)
        else: 
            resolution = resolution * 4

        if gmsh.isInitialized(): 
            gmsh.finalize()
        gmsh.initialize()

        # Erstelle ein neues Modell

        modelName = "sinerot"+str(periode)

        gmsh.model.add(modelName)

        lc = 1  # Charakteristische Länge, scheint egal zu sein, wenn unten meshmin- und meshmaxsize beschrieben sind 

        center_point = []

        # Kreispositionen
        # von welchem Punkt an geht ein neuer Kreis los? 
        von = []
        for i in range(0, len(crc_geo)):
            von.append(sum(crc_geo[0:i]))

        vertices_geo = np.column_stack((x_geo.ravel(), y_geo.ravel(), z_geo.ravel()))
        for i in range(len(vertices_geo)): 
            gmsh.model.geo.addPoint(vertices_geo[i][0],vertices_geo[i][1],vertices_geo[i][2], lc, i)

        # checken, ob x_btm ne Liste ist, dann gibt es mehrere Grenzschichten / Böden
        # vereinfacht die Bodensurfaces, nimmt nicht alle Punkte = kleinere Dateigröße
        if isinstance(x_btm[0], np.ndarray): 
            vertices_btm = np.array([None]*len(x_btm))
            for i in range(len(x_btm)): 
                index = von[-1]
                vertices_btm[i] = np.column_stack((x_btm[i][index:].ravel(), y_btm[i][index:].ravel(), z_btm[i][index:].ravel()))

                for ii in range(len(vertices_btm[i])): 
                    gmsh.model.geo.addPoint(vertices_btm[i][ii][0],vertices_btm[i][ii][1],vertices_btm[i][ii][2], lc, ii+len(vertices_geo)+len(vertices_btm[0])*(i))
        
            # Punkte im Zentren der Böden 
            for iii in range(len(x_btm)): 
                gmsh.model.geo.addPoint(0,0,vertices_btm[iii][0][2], lc, ii+len(vertices_geo)+len(vertices_btm[0])*(i)+iii+1)
                center_point.append(ii+len(vertices_geo)+len(vertices_btm[0])*(i)+iii+1)

            bottom = [[] for _ in range(len(x_btm))]
            walls = [[] for _ in range(len(x_btm))]

        # nur ein Boden
        else: 
            index = von[-1]
            vertices_btm = np.column_stack((x_btm[index:].ravel(), y_btm[index:].ravel(), z_btm[index:].ravel()))

            for i in range(len(vertices_btm)): 
                gmsh.model.geo.addPoint(vertices_btm[i][0],vertices_btm[i][1],vertices_btm[i][2], lc, i+len(vertices_geo))
            
            # Zentrum Boden 
            center_point.append(i+len(vertices_geo)+1)
            gmsh.model.geo.addPoint(0,0,vertices_btm[0][2], lc, center_point[0])

            bottom = []
            walls = []
        
        gmsh_surface_tag = 1
        gmsh_curve_tag = 1
        volume_tag = 1

        # alle Volumen
        volume_array = []

        # Physical Group Indices
        up = []

        jump_checker = False

        #for j in range(1, 2):
        for j in range(0, resolution-1):            # j ist immer der innere Ring
            if j == 0:                              # Spezialfall: Zentrum
                a = 0
                b = len(vertices_geo)

            else:
                a = von[j]                           # Punkt des vorherigen Kreises oben ---> mit i_1%2 == 0 iterieren, damit es sich jedes zweite mal ändert!!!
                b = von[j]+len(vertices_geo)                  # Punkt des vorherigen Kreises unten 

            i_3 = 1

            for i_1 in range(crc_geo[j+1]):             # Triangles pro Ring erzeugen

                # sobald ein Punkt des äußeren Kreises näher an einem Punkt des inneren Kreises dran ist, muss mit diesem verbunden werden 
                # phi_arr[:][1] ==> Inkrementierung des Winkels mit jedem neuen Punkt
                # phi_arr[j][:] + phi_arr[:][1]/2 ==> liegt genau zwischen zwei Punkten, daher Abfragebedingung
                #print(j)
                if (i_3 < len(phi_arr_geo[j])) and (j != 0) and (phi_arr_geo[j+1][i_1] > (phi_arr_geo[j][i_3]+phi_arr_geo[j][1]/5)):

                    a+=1
                    b+=1
                    i_3+=1
                    jump_checker = True

                c = von[j+1] + i_1                 # aktueller Kreis oben, Punkt 1
                d = c + 1                          # aktueller Kreis oben, Punkt 2

                #e = c+len(vertices_btm)             # aktueller Kreis unten, Punkt 1
                #f = e + 1                           # aktueller Kreis unten, Punkt 2

                h = (von[j+1]+crc_geo[j+1]*2)-1-crc_geo[j+1]    # Maximalwert des Kreises oben
                #g = h+len(vertices_btm[0])                        # Maximalwert des Kreises unten

                if i_1 ==0:
                    gmsh.model.geo.addLine(h, c ,gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(c, a ,gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(a, h ,gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                    gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                    up.append(gmsh_curve_tag)
                    gmsh_curve_tag+=1

                    if(j != 0):
                        gmsh.model.geo.addLine(h, a ,gmsh_surface_tag)
                        gmsh_surface_tag+=1
                        gmsh.model.geo.addLine(a, a+crc_geo[j]-1 ,gmsh_surface_tag)
                        gmsh_surface_tag+=1
                        gmsh.model.geo.addLine(a+crc_geo[j]-1, h ,gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                        up.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1

                    if(j > 3):      #Exception für zwei fehlende Dreiecke nebeneinander
                        gmsh.model.geo.addLine(h, a+crc_geo[j]-1 ,gmsh_surface_tag)
                        gmsh_surface_tag+=1
                        gmsh.model.geo.addLine(a+crc_geo[j]-1, a+crc_geo[j]-2 ,gmsh_surface_tag)
                        gmsh_surface_tag+=1
                        gmsh.model.geo.addLine(a+crc_geo[j]-2, h ,gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                        up.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1
                
                else:                 # reiner Increment um den Kreis
                    gmsh.model.geo.addLine(c-1, d-1 ,gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(d-1, a ,gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(a, c-1 ,gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                    gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                    up.append(gmsh_curve_tag)
                    gmsh_curve_tag+=1

                    if(jump_checker): 
                        gmsh.model.geo.addLine(a-1, c-1 ,gmsh_surface_tag)
                        gmsh_surface_tag+=1
                        gmsh.model.geo.addLine(c-1, a ,gmsh_surface_tag)
                        gmsh_surface_tag+=1
                        gmsh.model.geo.addLine(a, a-1 ,gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                        up.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1

                        jump_checker = False
                if j == resolution-2:                                               # Wände
                    if i_1 ==0:
                        if isinstance(x_btm[0], np.ndarray):
                            e = c+len(vertices_btm[0])             # aktueller Kreis unten, Punkt 1
                            f = e + 1                           # aktueller Kreis unten, Punkt 2
                            g = h+len(vertices_btm[0])
                            for iii in range(len(x_btm)): 
                                gmsh.model.geo.addLine(g+(iii*len(vertices_btm[iii])), e+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1
                                gmsh.model.geo.addLine(e+(iii*len(vertices_btm[iii])), h+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1
                                gmsh.model.geo.addLine(h+(iii*len(vertices_btm[iii])), g+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1

                                gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                                gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                                walls[iii].append(gmsh_curve_tag)
                                gmsh_curve_tag+=1

                                gmsh.model.geo.addLine(e+(iii*len(vertices_btm[iii])), c+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1
                                gmsh.model.geo.addLine(c+(iii*len(vertices_btm[iii])), h+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1
                                gmsh.model.geo.addLine(h+(iii*len(vertices_btm[iii])), e+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1

                                gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                                gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                                walls[iii].append(gmsh_curve_tag)
                                gmsh_curve_tag+=1
                        else:
                            e = c+len(vertices_btm)             # aktueller Kreis unten, Punkt 1
                            f = e + 1                           # aktueller Kreis unten, Punkt 2
                            g = h+len(vertices_btm)

                            gmsh.model.geo.addLine(g, e ,gmsh_surface_tag)
                            gmsh_surface_tag+=1
                            gmsh.model.geo.addLine(e, h ,gmsh_surface_tag)
                            gmsh_surface_tag+=1
                            gmsh.model.geo.addLine(h, g ,gmsh_surface_tag)
                            gmsh_surface_tag+=1

                            gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                            gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                            walls.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1

                            gmsh.model.geo.addLine(e, c ,gmsh_surface_tag)
                            gmsh_surface_tag+=1
                            gmsh.model.geo.addLine(c, h ,gmsh_surface_tag)
                            gmsh_surface_tag+=1
                            gmsh.model.geo.addLine(h, e ,gmsh_surface_tag)
                            gmsh_surface_tag+=1

                            gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                            gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                            walls.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1

                    else: 
                        if isinstance(x_btm[0], np.ndarray):
                            e = c+len(vertices_btm[0])             # aktueller Kreis unten, Punkt 1
                            f = e + 1                           # aktueller Kreis unten, Punkt 2
                            g = h+len(vertices_btm[0])
                            for iii in range(len(x_btm)): 
                                gmsh.model.geo.addLine(e-1+(iii*len(vertices_btm[iii])), f-1+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1
                                gmsh.model.geo.addLine(f-1+(iii*len(vertices_btm[iii])), c-1+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1
                                gmsh.model.geo.addLine(c-1+(iii*len(vertices_btm[iii])), e-1+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1

                                gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                                gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                                walls[iii].append(gmsh_curve_tag)
                                gmsh_curve_tag+=1

                                gmsh.model.geo.addLine(f-1+(iii*len(vertices_btm[iii])), d-1+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1
                                gmsh.model.geo.addLine(d-1+(iii*len(vertices_btm[iii])), c-1+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1
                                gmsh.model.geo.addLine(c-1+(iii*len(vertices_btm[iii])), f-1+(iii*len(vertices_btm[iii])) ,gmsh_surface_tag)
                                gmsh_surface_tag+=1

                                gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                                gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                                walls[iii].append(gmsh_curve_tag)
                                gmsh_curve_tag+=1
                        else:
                            e = c+len(vertices_btm)             # aktueller Kreis unten, Punkt 1
                            f = e + 1                           # aktueller Kreis unten, Punkt 2
                            g = h+len(vertices_btm)
                            gmsh.model.geo.addLine(e-1, f-1 ,gmsh_surface_tag)
                            gmsh_surface_tag+=1
                            gmsh.model.geo.addLine(f-1, c-1 ,gmsh_surface_tag)
                            gmsh_surface_tag+=1
                            gmsh.model.geo.addLine(c-1, e-1 ,gmsh_surface_tag)
                            gmsh_surface_tag+=1

                            gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                            gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                            walls.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1

                            gmsh.model.geo.addLine(f-1, d-1 ,gmsh_surface_tag)
                            gmsh_surface_tag+=1
                            gmsh.model.geo.addLine(d-1, c-1 ,gmsh_surface_tag)
                            gmsh_surface_tag+=1
                            gmsh.model.geo.addLine(c-1, f-1 ,gmsh_surface_tag)
                            gmsh_surface_tag+=1

                            gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                            gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                            walls.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1

        # # Böden als Tortenslices (reduce mesh file size)
        if isinstance(x_btm[0], np.ndarray):
            var = len(x_btm)
        else:
            var = 1

        for iii in range(var-1, var):
            for i in range(crc_geo[-1]): 
                if i == crc_geo[-1]-1: 
                    gmsh.model.geo.addLine(len(vertices_geo)+len(vertices_btm[0])*iii+i, len(vertices_geo)+len(vertices_btm[0])*iii ,gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(len(vertices_geo)+len(vertices_btm[0])*iii, center_point[iii] ,gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(center_point[iii], len(vertices_geo)+len(vertices_btm[0])*iii+i ,gmsh_surface_tag)
                    gmsh_surface_tag+=1   

                    gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                    gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag) 
                    if isinstance(x_btm[0], np.ndarray):
                        bottom[iii].append(gmsh_curve_tag)
                    else: 
                        bottom.append(gmsh_curve_tag)
                    gmsh_curve_tag+=1      
                else:
                    gmsh.model.geo.addLine(len(vertices_geo)+len(vertices_btm[0])*iii+i, len(vertices_geo)+len(vertices_btm[0])*iii+i+1 ,gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(len(vertices_geo)+len(vertices_btm[0])*iii+i+1,center_point[iii] ,gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(center_point[iii],len(vertices_geo)+len(vertices_btm[0])*iii+i ,gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    gmsh.model.geo.addCurveLoop([gmsh_surface_tag-3, gmsh_surface_tag-2, gmsh_surface_tag-1], gmsh_curve_tag)
                    gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag) 
                    if isinstance(x_btm[0], np.ndarray):
                        bottom[iii].append(gmsh_curve_tag)
                    else: 
                        bottom.append(gmsh_curve_tag)
                    gmsh_curve_tag+=1  

        # # Erstelle das Volumen
        if isinstance(x_btm[0], np.ndarray):
            walls_array=[]
            for iii in range(len(x_btm)): 
                walls_array+=walls[iii]
            surface_array = up+walls_array+bottom[len(x_btm)-1]

            gmsh.model.geo.addSurfaceLoop(surface_array, volume_tag)
            gmsh.model.geo.addVolume([volume_tag], volume_tag)
            volume_array.append(volume_tag)
            volume_tag+=1

            # # Synchronisiere die CAD-Entitäten mit dem Gmsh-Modell
            gmsh.model.geo.removeAllDuplicates()
            gmsh.model.geo.synchronize()

            print("Nachbarn von Fläche 1: "+ str(gmsh.model.getAdjacencies(2, 1)))

            gmsh.model.add_physical_group(2, walls_array, 1)
            #gmsh.model.add_physical_group(2, walls_array_mod, 1)
            gmsh.model.setPhysicalName(2, 1, "Walls")

            # gmsh.model.add_physical_group(2, walls[0], 5)
            # gmsh.model.setPhysicalName(2, 5, "Walls unten")

            # eine Physical Group für Geo erstellen
            gmsh.model.add_physical_group(2, up, 2)
            gmsh.model.setPhysicalName(2, 2, "Up")

            # Abschnitte erstellen
            # print("len up"+str(len(up)))
            # for i in range(len(up)): 
            #     gmsh.model.add_physical_group(2, [up[i]], 2+i)
            #     gmsh.model.setPhysicalName(2, 2+i, "Up "+str(i))

            # Abschnitte 
            # gmsh.model.add_physical_group(2, bottom[len(bottom)-1], 3+i)
            # gmsh.model.setPhysicalName(2, 3+i, "Bottom")

            # eine Physical Group für Geo 
            gmsh.model.add_physical_group(2, bottom[len(bottom)-1], 3)
            gmsh.model.setPhysicalName(2, 3, "Bottom")
            
            gmsh.model.add_physical_group(3, volume_array, 4)
            gmsh.model.setPhysicalName(3, 4, "Volume")
        else:
            surface_array = up+walls+bottom
            #print(surface_array)

            gmsh.model.geo.addSurfaceLoop(surface_array, volume_tag)
            gmsh.model.geo.addVolume([volume_tag], volume_tag)

            # # Synchronisiere die CAD-Entitäten mit dem Gmsh-Modell
            gmsh.model.geo.synchronize()

            gmsh.model.add_physical_group(2, walls, 1)
            gmsh.model.setPhysicalName(2, 1, "Walls")

            print("len up"+str(len(up)))
            for i in range(len(up)): 
                gmsh.model.add_physical_group(2, up[i], 2)
                gmsh.model.setPhysicalName(2, 2, "Up "+str(i))

            gmsh.model.add_physical_group(2, bottom, 3)
            gmsh.model.setPhysicalName(2, 3, "Bottom")

            gmsh.model.add_physical_group(3, [1], 4)
            gmsh.model.setPhysicalName(3, 4, "Volume")

        # gmsh.option.setNumber("Mesh.Algorithm",1)  # Try different algorithms (1 to 7)
        #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1) # Muss 1 sein, wenn gmsh eigene Dreiecke baut, gibt es Richtungsfehler!
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)

        # Generiere das 3D-Mesh
        gmsh.model.mesh.generate(3)
        if order > 1: 
            gmsh.model.mesh.setOrder(order)
            gmsh.model.mesh.optimize("HighOrder")
    
        # Wenn du die GUI verwenden möchtest, um das Mesh zu visualisieren
        if '-nopopup' not in sys.argv and gmsh_gui == True:
            gmsh.fltk.run()

        # Speichere das Mesh in einer Datei
        gmsh.write(str(name)+".msh")
        if stl == True: 
            gmsh.write(str(name)+".stl")
        gmsh.finalize()

    # Korrigiert sämtliche Duplikate und unzusammenhängende Surfaces: 
    def material4(self, x_geo, y_geo, z_geo, crc_geo, phi_arr_geo, x_btm, y_btm, z_btm, crc_btm, phi_arr_btm, **kwargs): 
        # GMSH Generation von meinem Meshcode! Hurra! Es läuft! 
        if len(crc_btm) != len(crc_geo) or len(phi_arr_geo) != len(phi_arr_btm): 
            return "Error: crc and phi_arr don't have same length."

        name = kwargs.get('saveas', 'default')
        gmsh_gui = kwargs.get('gmsh_gui', True)
        periode = kwargs.get('period', 1)
        resolution = kwargs.get('resolution', 10)
        komp = kwargs.get('counterpart', False)
        # twodom = kwargs.get('two_domains', False)
        rec = kwargs.get('rec', False)
        order = kwargs.get('mesh_order', 3)

        stl = kwargs.get('generate_stl', False)
        msh = kwargs.get('generate_msh', True)

        lc = kwargs.get('lc', 0.01)
        lc_geo = kwargs.get('lc_geo', 0.01)

        if komp: 
            kvar = -1
        else:
            kvar = 1

        if rec: 
            resolution = len(crc_geo)
        else: 
            resolution = resolution * 4

        if gmsh.isInitialized(): 
            gmsh.finalize()
        gmsh.initialize()

        # Erstelle ein neues Modell
        modelName = "sinerot"+str(periode)
        gmsh.model.add(modelName)

        #lc = 0.01  # Charakteristische Länge, scheint egal zu sein, wenn unten meshmin- und meshmaxsize beschrieben sind 

        # Kreispositionen
        # von welchem Punkt an geht ein neuer Kreis los? 
        von = []
        for i in range(0, len(crc_geo)):
            von.append(sum(crc_geo[0:i]))

        vertices_geo = np.column_stack((x_geo.ravel(), y_geo.ravel(), z_geo.ravel()))
        for i in range(len(vertices_geo)): 
            # Tulip 1 Hollow Code, damit man es drucken kann: 
            # if i == 7: 
            #     gmsh.model.geo.addPoint(vertices_geo[i][0],vertices_geo[i][1],-0.05, lc_geo, i)
            # else: 
            #     gmsh.model.geo.addPoint(vertices_geo[i][0],vertices_geo[i][1],vertices_geo[i][2], lc_geo, i)
            gmsh.model.geo.addPoint(vertices_geo[i][0],vertices_geo[i][1],vertices_geo[i][2], lc_geo, i)

            bottom = [[] for _ in range(len(x_btm))]
            walls = [[] for _ in range(len(x_btm))]

        # nur ein Boden
        center_point=[]
        index = von[-1]
        vertices_btm = np.column_stack((x_btm.ravel(), y_btm.ravel(), z_btm.ravel()))

        for i in range(len(vertices_btm)): 
            gmsh.model.geo.addPoint(vertices_btm[i][0],vertices_btm[i][1],vertices_btm[i][2], lc, i+len(vertices_geo))
        
        # Zentrum Boden 
        center_point.append(i+len(vertices_geo)+1)
        gmsh.model.geo.addPoint(0,0,vertices_btm[0][2], lc, center_point[0])

        bottom = []
        walls = []
        
        gmsh_surface_tag = 1
        gmsh_curve_tag = 1
        volume_tag = 1

        # Physical Group Indices
        up = []

        jump_checker = False

        # Variable für Linienermittlung bei Flächenerstellung
        var = 1
        line_var = 1
        line_var_arr = []

        i_0_arr = []
        i_0_arr_2 = []
        j_ungleich_0 = []
        j_groesser_3 = []
        anders_kreis = [[] for _ in range(resolution)]
        anders_ecke = [[] for _ in range(resolution)]
        jump_checker_true = [[] for _ in range(resolution)]

        var_arr = []

        gmsh.model.geo.synchronize()
        gmsh.model.geo.removeAllDuplicates()

        # Linien Geo: 
        for j in range(0, resolution-1):            # j ist immer der innere Ring
            if j == 0:                              # Spezialfall: Zentrum
                a = 0
                b = len(vertices_geo)

            else:
                a = von[j]                           # Punkt des vorherigen Kreises oben ---> mit i_1%2 == 0 iterieren, damit es sich jedes zweite mal ändert!!!
                b = von[j]+len(vertices_geo)                  # Punkt des vorherigen Kreises unten 

            i_3 = 1

            for i_1 in range(crc_geo[j+1]):             # Triangles pro Ring erzeugen
            # for i_1 in range(0, 2): 

                if (i_3 < len(phi_arr_geo[j])) and (j != 0) and (phi_arr_geo[j+1][i_1] > (phi_arr_geo[j][i_3]+phi_arr_geo[j][1]/5)):
                    a+=1
                    b+=1
                    i_3+=1
                    jump_checker = True

                if i_1 ==0 and j>0 or jump_checker:
                    line_var += 3
                    line_var_arr.append(line_var)
                else:
                    line_var += 2
                    line_var_arr.append(line_var)

                # print(line_var_arr)
                c = von[j+1] + i_1                 # aktueller Kreis oben, Punkt 1
                d = c + 1                          # aktueller Kreis oben, Punkt 2

                h = (von[j+1]+crc_geo[j+1]*2)-1-crc_geo[j+1]    # Maximalwert des Kreises oben

                if i_1 ==0:
                    gmsh.model.geo.addLine(h, c ,gmsh_surface_tag)
                    i_0_arr.append(gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(c, a ,gmsh_surface_tag)
                    i_0_arr_2.append(gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    res = int(resolution/4)

                    var +=(res*2)+j*(res*3)
                    var_arr.append(var)

                    if(j != 0):
                        gmsh.model.geo.addLine(h, a ,gmsh_surface_tag)
                        j_ungleich_0.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.geo.addCurveLoop([gmsh_surface_tag-1, -(gmsh_surface_tag-2), -(gmsh_surface_tag-3)], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                        up.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1

                    if(j > 3):      #Exception für zwei fehlende Dreiecke nebeneinander
                        gmsh.model.geo.addLine(h, a+crc_geo[j]-1 ,gmsh_surface_tag)
                        j_groesser_3.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.geo.addCurveLoop([var_arr[j-2], -(gmsh_surface_tag-2), gmsh_surface_tag-1], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                        up.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1

                else:                 # reiner Increment um den Kreis
                    gmsh.model.geo.addLine(c-1, d-1 ,gmsh_surface_tag)
                    anders_kreis[j].append(gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(d-1, a ,gmsh_surface_tag)
                    anders_ecke[j].append(gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    if(jump_checker == False):
                        if j == 0: 
                            gmsh.model.geo.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), gmsh_surface_tag-3], gmsh_curve_tag)
                            gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                            up.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1 
                        if j == 1 and i_1 == 2:
                            gmsh.model.geo.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), (gmsh_surface_tag-3)], gmsh_curve_tag)
                            gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                            up.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1
                        if j > 0 and i_1 > 2:    
                            gmsh.model.geo.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), (gmsh_surface_tag-4)], gmsh_curve_tag)
                            gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                            up.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1                        

                    elif(jump_checker):
                        gmsh.model.geo.addLine(c-1, a ,gmsh_surface_tag)
                        jump_checker_true[j].append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.geo.addCurveLoop([-(gmsh_surface_tag-3), gmsh_surface_tag-1, -(gmsh_surface_tag-2)], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                        up.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1
                    
                        jump_checker = False

        # Flächen für Geo
        for j in range(resolution-2): 
            for i in range(len(anders_kreis[j])):
                if j == 0 and i== 0: 
                    gmsh.model.geo.addCurveLoop([-2,-1,res*2 ], gmsh_curve_tag)
                    gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                    up.append(gmsh_curve_tag)
                    gmsh_curve_tag+=1 

                for tri in range(len(anders_ecke[j+1])): # anders_ecke durchiterieren und gucken, was passt. Kein Bock mehr. 
                    try: 
                        gmsh.model.geo.addCurveLoop([anders_kreis[j][i],-jump_checker_true[j+1][i] ,anders_ecke[j+1][tri] ], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                        up.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1
                    except:
                        tri+=1

            gmsh.model.geo.addCurveLoop([-anders_ecke[j+1][0], -anders_kreis[j+1][0], i_0_arr_2[j+1]], gmsh_curve_tag)
            gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
            up.append(gmsh_curve_tag)
            gmsh_curve_tag+=1          
            if j < 3: 
                gmsh.model.geo.addCurveLoop([i_0_arr[j], -j_ungleich_0[j], anders_ecke[j+1][-1] ], gmsh_curve_tag)
                gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                up.append(gmsh_curve_tag)
                gmsh_curve_tag+=1       
            else: 
                gmsh.model.geo.addCurveLoop([anders_kreis[j][-1], -j_groesser_3[j-3], anders_ecke[j+1][-1]], gmsh_curve_tag)
                gmsh.model.geo.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                up.append(gmsh_curve_tag)
                gmsh_curve_tag+=1

        ######################################################
        # Variable für Linienermittlung bei Flächenerstellung
        var = 1
        line_var = 1
        line_var_arr = []

        i_0_arr_btm = []
        i_0_arr_2 = []
        j_ungleich_0 = []
        j_groesser_3 = []
        anders_kreis_btm = [[] for _ in range(resolution)]
        anders_ecke = [[] for _ in range(resolution)]
        jump_checker_true = [[] for _ in range(resolution)]

        var_arr = []

        # Linien Bottom: 
        for j in range(0, resolution-1):            # j ist immer der innere Ring
            if j == 0:                              # Spezialfall: Zentrum
                a = 0+len(vertices_geo)
                b = len(vertices_geo)+len(vertices_btm)

            else:
                a = von[j]+len(vertices_geo)                           # Punkt des vorherigen Kreises oben ---> mit i_1%2 == 0 iterieren, damit es sich jedes zweite mal ändert!!!
                b = von[j]+len(vertices_geo)+len(vertices_btm)                  # Punkt des vorherigen Kreises unten 

            i_3 = 1

            for i_1 in range(crc_geo[j+1]):             # Triangles pro Ring erzeugen

                if (i_3 < len(phi_arr_geo[j])) and (j != 0) and (phi_arr_geo[j+1][i_1] > (phi_arr_geo[j][i_3]+phi_arr_geo[j][1]/5)):
                    a+=1
                    b+=1
                    i_3+=1
                    jump_checker = True

                if i_1 ==0 and j>0 or jump_checker:
                    line_var += 3
                    line_var_arr.append(line_var)
                else:
                    line_var += 2
                    line_var_arr.append(line_var)

                # print(line_var_arr)
                c = von[j+1] + i_1 + len(vertices_geo)                # aktueller Kreis oben, Punkt 1
                d = c + 1                          # aktueller Kreis oben, Punkt 2

                h = (von[j+1]+crc_geo[j+1]*2)-1-crc_geo[j+1] + len(vertices_geo)    # Maximalwert des Kreises oben

                if i_1 ==0:
                    gmsh.model.geo.addLine(h, c ,gmsh_surface_tag)
                    i_0_arr_btm.append(gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(c, a ,gmsh_surface_tag)
                    i_0_arr_2.append(gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    res = int(resolution/4)

                    var +=(res*2)+j*(res*3)
                    var_arr.append(var + (gmsh_surface_tag)-38)

                    if(j != 0):
                        gmsh.model.geo.addLine(h, a ,gmsh_surface_tag)
                        j_ungleich_0.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.geo.addCurveLoop([gmsh_surface_tag-1, -(gmsh_surface_tag-2), -(gmsh_surface_tag-3)], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                        bottom.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1

                    if(j > 3):      #Exception für zwei fehlende Dreiecke nebeneinander
                        gmsh.model.geo.addLine(h, a+crc_geo[j]-1 ,gmsh_surface_tag)
                        j_groesser_3.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.geo.addCurveLoop([i_0_arr_btm[j-1], -(gmsh_surface_tag-2), gmsh_surface_tag-1], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                        bottom.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1

                else:                 # reiner Increment um den Kreis
                    gmsh.model.geo.addLine(c-1, d-1 ,gmsh_surface_tag)
                    anders_kreis_btm[j].append(gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.geo.addLine(d-1, a ,gmsh_surface_tag)
                    anders_ecke[j].append(gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    if(jump_checker == False):
                        if j == 0: 
                            gmsh.model.geo.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), gmsh_surface_tag-3], gmsh_curve_tag)
                            gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                            bottom.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1 
                        if j == 1 and i_1 == 2:
                            gmsh.model.geo.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), (gmsh_surface_tag-3)], gmsh_curve_tag)
                            gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                            bottom.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1
                        if j > 0 and i_1 > 2:    
                            gmsh.model.geo.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), (gmsh_surface_tag-4)], gmsh_curve_tag)
                            gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                            bottom.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1                        

                    elif(jump_checker):
                        gmsh.model.geo.addLine(c-1, a ,gmsh_surface_tag)
                        jump_checker_true[j].append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.geo.addCurveLoop([-(gmsh_surface_tag-3), gmsh_surface_tag-1, -(gmsh_surface_tag-2)], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                        bottom.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1
                    
                        jump_checker = False

        # # Flächen für Bottom
        for j in range(resolution-2): 
            for i in range(len(anders_kreis[j])):
                if j == 0 and i== 0: 
                    # print("i0 bottom "+str(i_0_arr_btm))
                    # print("circ bottom "+str(anders_kreis_btm))
                    # print("j > 3 "+str(j_groesser_3))
                    # print("i_0 arr 2 "+str(anders_ecke))
                    gmsh.model.geo.addCurveLoop([-i_0_arr_2[0],-i_0_arr_btm[0],anders_ecke[0][-1] ], gmsh_curve_tag)
                    gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                    bottom.append(gmsh_curve_tag)
                    gmsh_curve_tag+=1 

                for tri in range(len(anders_ecke[j+1])): # anders_ecke durchiterieren und gucken, was passt. Kein Bock mehr. 
                    try: 
                        gmsh.model.geo.addCurveLoop([anders_kreis_btm[j][i],-jump_checker_true[j+1][i] ,anders_ecke[j+1][tri] ], gmsh_curve_tag)
                        gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                        bottom.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1
                    except:
                        tri+=1

            gmsh.model.geo.addCurveLoop([-anders_ecke[j+1][0], -anders_kreis_btm[j+1][0], i_0_arr_2[j+1]], gmsh_curve_tag)
            gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
            bottom.append(gmsh_curve_tag)
            gmsh_curve_tag+=1          
            if j < 3: 
                gmsh.model.geo.addCurveLoop([i_0_arr_btm[j], -j_ungleich_0[j], anders_ecke[j+1][-1] ], gmsh_curve_tag)
                gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                bottom.append(gmsh_curve_tag)
                gmsh_curve_tag+=1       
            else: 
                gmsh.model.geo.addCurveLoop([anders_kreis_btm[j][-1], -j_groesser_3[j-3], anders_ecke[j+1][-1]], gmsh_curve_tag)
                gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                bottom.append(gmsh_curve_tag)
                gmsh_curve_tag+=1

        schraeg = []
        gerade = []

        # Linien und Flächen Wände: 
        for j in range(0, resolution-1):            # j ist immer der innere Ring
            for i_1 in range(crc_geo[j+1]):             # Triangles pro Ring erzeugen
                c = von[j+1] + i_1                 # aktueller Kreis oben, Punkt 1
                h = (von[j+1]+crc_geo[j+1]*2)-1-crc_geo[j+1]    # Maximalwert des Kreises oben
                if j == resolution-2:
                    if i_1 ==0:
                        e = c+len(vertices_btm)             # aktueller Kreis unten, Punkt 1
                        f = e + 1                           # aktueller Kreis unten, Punkt 2
                        g = h+len(vertices_btm)
                        # erste und letzte gerade wand
                        gmsh.model.geo.addLine(h, g ,gmsh_surface_tag)
                        gerade.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1
                        gmsh.model.geo.addLine(e, c ,gmsh_surface_tag)
                        gerade.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1
                    else: 
                        e = c+len(vertices_btm)             # aktueller Kreis unten, Punkt 1
                        f = e + 1                           # aktueller Kreis unten, Punkt 2
                        g = h+len(vertices_btm)
                        if i_1 != crc_geo[j+1]-1: 
                            gmsh.model.geo.addLine(c, f-1 ,gmsh_surface_tag)
                            gerade.append(gmsh_surface_tag)
                            gmsh_surface_tag+=1

        for i_1 in range(len(anders_kreis[-2])):
            try: 
                gmsh.model.geo.addCurveLoop([-anders_kreis[-2][i_1], gerade[i_1+1],anders_kreis_btm[-2][i_1], -gerade[i_1+2]], gmsh_curve_tag)
                gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
                walls.append(gmsh_curve_tag)
                gmsh_curve_tag+=1
            except: 
                continue

        gmsh.model.geo.addCurveLoop([-anders_kreis[-2][0], -gerade[1],anders_kreis_btm[-2][0], -gerade[2]], gmsh_curve_tag)
        gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
        walls.append(gmsh_curve_tag)
        gmsh_curve_tag+=1

        gmsh.model.geo.addCurveLoop([-anders_kreis[-2][-1], gerade[-1],anders_kreis_btm[-2][-1], -gerade[0]], gmsh_curve_tag)
        gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
        walls.append(gmsh_curve_tag)
        gmsh_curve_tag+=1

        gmsh.model.geo.addCurveLoop([-i_0_arr[-1], gerade[0], i_0_arr_btm[-1], gerade[1]], gmsh_curve_tag)
        gmsh.model.geo.addPlaneSurface([-kvar*gmsh_curve_tag], gmsh_curve_tag)
        walls.append(gmsh_curve_tag)
        gmsh_curve_tag+=1

        # # Erstelle das Volumen
        surface_array = up+walls+bottom
        # gmsh.model.geo.synchronize()
        # flchen_vorher = gmsh.model.getEntities(0)
        # gmsh.model.geo.removeAllDuplicates()
        # gmsh.model.geo.synchronize()
        # flchen_nachher = gmsh.model.getEntities(0)
        # print("vor"+str(len(flchen_vorher)))
        # print("nach"+str(len(flchen_nachher)))
        # print("vor"+str(flchen_vorher))
        # print("nach"+str(flchen_nachher))

        gmsh.model.geo.addSurfaceLoop(surface_array, volume_tag)
        gmsh.model.geo.addVolume([volume_tag], volume_tag)

        # # # Synchronisiere die CAD-Entitäten mit dem Gmsh-Modell
        gmsh.model.geo.synchronize()

        gmsh.model.add_physical_group(2, walls, 1)
        gmsh.model.setPhysicalName(2, 1, "Walls")

        gmsh.model.add_physical_group(2, up, 2)
        gmsh.model.setPhysicalName(2, 2, "Up")

        gmsh.model.add_physical_group(2, bottom, 3)
        gmsh.model.setPhysicalName(2, 3, "Bottom")

        gmsh.model.add_physical_group(3, [1], 4)
        gmsh.model.setPhysicalName(3, 4, "Volume")

        # Synchronisiere die CAD-Entitäten mit dem Gmsh-Modell
        # gmsh.model.geo.removeAllDuplicates()
        gmsh.model.geo.synchronize()

        # gmsh.option.setNumber("Mesh.Algorithm",7)  # Try different algorithms (1 to 7)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1) # Muss 1 sein, wenn gmsh eigene Dreiecke baut, gibt es Richtungsfehler!
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)

        # Generiere das 3D-Mesh
        gmsh.model.mesh.generate(3)
        if order > 1: 
            gmsh.model.mesh.setOrder(order)
            gmsh.model.mesh.optimize("HighOrder")
    
        # Wenn du die GUI verwenden möchtest, um das Mesh zu visualisieren
        if '-nopopup' not in sys.argv and gmsh_gui == True:
            gmsh.fltk.run()

        # Speichere das Mesh in einer Datei
        if msh == True: 
            gmsh.write(str(name)+".msh")
        if stl == True: 
            gmsh.write(str(name)+".stl")
        gmsh.finalize()           
    
    # Material 4 mit occ für Step Export
    def material4_occ(self, x_geo, y_geo, z_geo, crc_geo, phi_arr_geo, x_btm, y_btm, z_btm, crc_btm, phi_arr_btm, **kwargs): 
        # GMSH Generation von meinem Meshcode! Hurra! Es läuft! 
        if len(crc_btm) != len(crc_geo) or len(phi_arr_geo) != len(phi_arr_btm): 
            return "Error: crc and phi_arr don't have same length."

        name = kwargs.get('saveas', 'default')
        gmsh_gui = kwargs.get('gmsh_gui', True)
        periode = kwargs.get('period', 1)
        resolution = kwargs.get('resolution', 10)
        komp = kwargs.get('counterpart', False)
        # twodom = kwargs.get('two_domains', False)
        rec = kwargs.get('rec', False)
        order = kwargs.get('mesh_order', 3)

        stl = kwargs.get('generate_stl', False)
        msh = kwargs.get('generate_msh', True)
        geo = kwargs.get('generate_step', False)

        lc = kwargs.get('lc', 0.01)
        lc_geo = kwargs.get('lc_geo', 0.01)

        if komp: 
            kvar = -1
        else:
            kvar = 1

        if rec: 
            resolution = len(crc_geo)
        else: 
            resolution = resolution * 4

        if gmsh.isInitialized(): 
            gmsh.finalize()
        gmsh.initialize()

        # Erstelle ein neues Modell
        modelName = "sinerot"+str(periode)
        gmsh.model.add(modelName)

        #lc = 0.01  # Charakteristische Länge, scheint egal zu sein, wenn unten meshmin- und meshmaxsize beschrieben sind 

        # Kreispositionen
        # von welchem Punkt an geht ein neuer Kreis los? 
        von = []
        for i in range(0, len(crc_geo)):
            von.append(sum(crc_geo[0:i]))

        vertices_geo = np.column_stack((x_geo.ravel(), y_geo.ravel(), z_geo.ravel()))
        for i in range(len(vertices_geo)): 
            gmsh.model.occ.addPoint(vertices_geo[i][0],vertices_geo[i][1],vertices_geo[i][2], lc_geo, i)

            bottom = [[] for _ in range(len(x_btm))]
            walls = [[] for _ in range(len(x_btm))]

        # nur ein Boden
        center_point=[]
        index = von[-1]
        vertices_btm = np.column_stack((x_btm.ravel(), y_btm.ravel(), z_btm.ravel()))

        for i in range(len(vertices_btm)): 
            gmsh.model.occ.addPoint(vertices_btm[i][0],vertices_btm[i][1],vertices_btm[i][2], lc, i+len(vertices_geo))
        
        # Zentrum Boden 
        center_point.append(i+len(vertices_geo)+1)
        gmsh.model.occ.addPoint(0,0,vertices_btm[0][2], lc, center_point[0])

        bottom = []
        walls = []
        
        gmsh_surface_tag = 1
        gmsh_curve_tag = 1
        volume_tag = 1

        # Physical Group Indices
        up = []

        jump_checker = False

        # Variable für Linienermittlung bei Flächenerstellung
        var = 1
        line_var = 1
        line_var_arr = []

        i_0_arr = []
        i_0_arr_2 = []
        j_ungleich_0 = []
        j_groesser_3 = []
        anders_kreis = [[] for _ in range(resolution)]
        anders_ecke = [[] for _ in range(resolution)]
        jump_checker_true = [[] for _ in range(resolution)]

        var_arr = []

        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()

        # Linien Geo: 
        for j in range(0, resolution-1):            # j ist immer der innere Ring
            if j == 0:                              # Spezialfall: Zentrum
                a = 0
                b = len(vertices_geo)

            else:
                a = von[j]                           # Punkt des vorherigen Kreises oben ---> mit i_1%2 == 0 iterieren, damit es sich jedes zweite mal ändert!!!
                b = von[j]+len(vertices_geo)                  # Punkt des vorherigen Kreises unten 

            i_3 = 1

            for i_1 in range(crc_geo[j+1]):             # Triangles pro Ring erzeugen
            # for i_1 in range(0, 2): 

                if (i_3 < len(phi_arr_geo[j])) and (j != 0) and (phi_arr_geo[j+1][i_1] > (phi_arr_geo[j][i_3]+phi_arr_geo[j][1]/5)):
                    a+=1
                    b+=1
                    i_3+=1
                    jump_checker = True

                if i_1 ==0 and j>0 or jump_checker:
                    line_var += 3
                    line_var_arr.append(line_var)
                else:
                    line_var += 2
                    line_var_arr.append(line_var)

                # print(line_var_arr)
                c = von[j+1] + i_1                 # aktueller Kreis oben, Punkt 1
                d = c + 1                          # aktueller Kreis oben, Punkt 2

                h = (von[j+1]+crc_geo[j+1]*2)-1-crc_geo[j+1]    # Maximalwert des Kreises oben

                if i_1 ==0:
                    gmsh.model.occ.addLine(h, c ,gmsh_surface_tag)
                    i_0_arr.append(gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.occ.addLine(c, a ,gmsh_surface_tag)
                    i_0_arr_2.append(gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    res = int(resolution/4)

                    var +=(res*2)+j*(res*3)
                    var_arr.append(var)

                    if(j != 0):
                        gmsh.model.occ.addLine(h, a ,gmsh_surface_tag)
                        j_ungleich_0.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1
                        try: 
                            gmsh.model.occ.addCurveLoop([gmsh_surface_tag-1, -(gmsh_surface_tag-2), -(gmsh_surface_tag-3)], gmsh_curve_tag)
                            gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                            up.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1
                        except:
                            gmsh.model.occ.addCurveLoop([gmsh_surface_tag-1, -(gmsh_surface_tag-2), -(gmsh_surface_tag-3)], gmsh_curve_tag+1)
                            gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                            up.append(gmsh_curve_tag+1)
                            gmsh_curve_tag+=2
                    if(j > 3):      #Exception für zwei fehlende Dreiecke nebeneinander
                        gmsh.model.occ.addLine(h, a+crc_geo[j]-1 ,gmsh_surface_tag)
                        j_groesser_3.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.occ.addCurveLoop([var_arr[j-2], -(gmsh_surface_tag-2), gmsh_surface_tag-1], gmsh_curve_tag+1)
                        gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                        up.append(gmsh_curve_tag+1)
                        gmsh_curve_tag+=2

                else:                 # reiner Increment um den Kreis
                    gmsh.model.occ.addLine(c-1, d-1 ,gmsh_surface_tag)
                    anders_kreis[j].append(gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.occ.addLine(d-1, a ,gmsh_surface_tag)
                    anders_ecke[j].append(gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    if(jump_checker == False):
                        if j == 0: 
                            try:
                                gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), gmsh_surface_tag-3], gmsh_curve_tag)
                                gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                                up.append(gmsh_curve_tag)
                                gmsh_curve_tag+=1 
                            except: 
                                gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), gmsh_surface_tag-3], gmsh_curve_tag+1)
                                gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                                up.append(gmsh_curve_tag+1)
                                gmsh_curve_tag+=2 

                        if j == 1 and i_1 == 2:
                            gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), (gmsh_surface_tag-3)], gmsh_curve_tag+1)
                            gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                            up.append(gmsh_curve_tag+1)
                            gmsh_curve_tag+=2
                        if j > 0 and i_1 > 2:
                            try:     
                                gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), (gmsh_surface_tag-4)], gmsh_curve_tag)
                                gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                                up.append(gmsh_curve_tag)
                                gmsh_curve_tag+=1
                            except:                         
                                gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), (gmsh_surface_tag-4)], gmsh_curve_tag+1)
                                gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                                up.append(gmsh_curve_tag+1)
                                gmsh_curve_tag+=2

                    elif(jump_checker):
                        gmsh.model.occ.addLine(c-1, a ,gmsh_surface_tag)
                        jump_checker_true[j].append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        try: 
                            gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-3), gmsh_surface_tag-1, -(gmsh_surface_tag-2)], gmsh_curve_tag)
                            gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                            up.append(gmsh_curve_tag)
                            gmsh_curve_tag+=1
                        except: 
                            gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-3), gmsh_surface_tag-1, -(gmsh_surface_tag-2)], gmsh_curve_tag+1)
                            gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                            up.append(gmsh_curve_tag+1)
                            gmsh_curve_tag+=2

                        jump_checker = False

        # Flächen für Geo
        for j in range(resolution-2): 
            for i in range(len(anders_kreis[j])):
                if j == 0 and i== 0: 
                    try: 
                        gmsh.model.occ.addCurveLoop([-2,-1,res*2 ], gmsh_curve_tag)
                        gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag], gmsh_curve_tag)
                        up.append(gmsh_curve_tag)
                        gmsh_curve_tag+=1 
                    except: 
                        gmsh.model.occ.addCurveLoop([-2,-1,res*2 ], gmsh_curve_tag+1)
                        gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                        up.append(gmsh_curve_tag+1)
                        gmsh_curve_tag+=2 

                for tri in range(len(anders_ecke[j+1])): # anders_ecke durchiterieren und gucken, was passt. Kein Bock mehr. 
                    try: 
                        gmsh.model.occ.addCurveLoop([anders_kreis[j][i],-jump_checker_true[j+1][i] ,anders_ecke[j+1][tri] ], gmsh_curve_tag+1)
                        gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                        up.append(gmsh_curve_tag+1)
                        gmsh_curve_tag+=2
                    except:
                        tri+=1

            gmsh.model.occ.addCurveLoop([-anders_ecke[j+1][0], -anders_kreis[j+1][0], i_0_arr_2[j+1]], gmsh_curve_tag+1)
            gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
            up.append(gmsh_curve_tag+1)
            gmsh_curve_tag+=2    
            if j < 3: 
                try: 
                    gmsh.model.occ.addCurveLoop([i_0_arr[j], -j_ungleich_0[j], anders_ecke[j+1][-1] ], gmsh_curve_tag+1)
                    gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                    up.append(gmsh_curve_tag+1)
                    gmsh_curve_tag+=2
                except: 
                    gmsh.model.occ.addCurveLoop([i_0_arr[j], -j_ungleich_0[j], anders_ecke[j+1][-1] ], gmsh_curve_tag+1)
                    gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                    up.append(gmsh_curve_tag+1)
                    gmsh_curve_tag+=2       
            else: 
                gmsh.model.occ.addCurveLoop([anders_kreis[j][-1], -j_groesser_3[j-3], anders_ecke[j+1][-1]], gmsh_curve_tag+1)
                gmsh.model.occ.addPlaneSurface([kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                up.append(gmsh_curve_tag+1)
                gmsh_curve_tag+=2

        ######################################################
        # Variable für Linienermittlung bei Flächenerstellung
        var = 1
        line_var = 1
        line_var_arr = []

        i_0_arr_btm = []
        i_0_arr_2 = []
        j_ungleich_0 = []
        j_groesser_3 = []
        anders_kreis_btm = [[] for _ in range(resolution)]
        anders_ecke = [[] for _ in range(resolution)]
        jump_checker_true = [[] for _ in range(resolution)]

        var_arr = []

        # Linien Bottom: 
        for j in range(0, resolution-1):            # j ist immer der innere Ring
            if j == 0:                              # Spezialfall: Zentrum
                a = 0+len(vertices_geo)
                b = len(vertices_geo)+len(vertices_btm)

            else:
                a = von[j]+len(vertices_geo)                           # Punkt des vorherigen Kreises oben ---> mit i_1%2 == 0 iterieren, damit es sich jedes zweite mal ändert!!!
                b = von[j]+len(vertices_geo)+len(vertices_btm)                  # Punkt des vorherigen Kreises unten 

            i_3 = 1

            for i_1 in range(crc_geo[j+1]):             # Triangles pro Ring erzeugen

                if (i_3 < len(phi_arr_geo[j])) and (j != 0) and (phi_arr_geo[j+1][i_1] > (phi_arr_geo[j][i_3]+phi_arr_geo[j][1]/5)):
                    a+=1
                    b+=1
                    i_3+=1
                    jump_checker = True

                if i_1 ==0 and j>0 or jump_checker:
                    line_var += 3
                    line_var_arr.append(line_var)
                else:
                    line_var += 2
                    line_var_arr.append(line_var)

                # print(line_var_arr)
                c = von[j+1] + i_1 + len(vertices_geo)                # aktueller Kreis oben, Punkt 1
                d = c + 1                          # aktueller Kreis oben, Punkt 2

                h = (von[j+1]+crc_geo[j+1]*2)-1-crc_geo[j+1] + len(vertices_geo)    # Maximalwert des Kreises oben

                if i_1 ==0:
                    gmsh.model.occ.addLine(h, c ,gmsh_surface_tag)
                    i_0_arr_btm.append(gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.occ.addLine(c, a ,gmsh_surface_tag)
                    i_0_arr_2.append(gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    res = int(resolution/4)

                    var +=(res*2)+j*(res*3)
                    var_arr.append(var + (gmsh_surface_tag)-38)

                    if(j != 0):
                        gmsh.model.occ.addLine(h, a ,gmsh_surface_tag)
                        j_ungleich_0.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.occ.addCurveLoop([gmsh_surface_tag-1, -(gmsh_surface_tag-2), -(gmsh_surface_tag-3)], gmsh_curve_tag+1)
                        gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                        bottom.append(gmsh_curve_tag+1)
                        gmsh_curve_tag+=2

                    if(j > 3):      #Exception für zwei fehlende Dreiecke nebeneinander
                        gmsh.model.occ.addLine(h, a+crc_geo[j]-1 ,gmsh_surface_tag)
                        j_groesser_3.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.occ.addCurveLoop([i_0_arr_btm[j-1], -(gmsh_surface_tag-2), gmsh_surface_tag-1], gmsh_curve_tag+1)
                        gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                        bottom.append(gmsh_curve_tag+1)
                        gmsh_curve_tag+=2

                else:                 # reiner Increment um den Kreis
                    gmsh.model.occ.addLine(c-1, d-1 ,gmsh_surface_tag)
                    anders_kreis_btm[j].append(gmsh_surface_tag)
                    gmsh_surface_tag+=1
                    gmsh.model.occ.addLine(d-1, a ,gmsh_surface_tag)
                    anders_ecke[j].append(gmsh_surface_tag)
                    gmsh_surface_tag+=1

                    if(jump_checker == False):
                        if j == 0: 
                            gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), gmsh_surface_tag-3], gmsh_curve_tag+1)
                            gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                            bottom.append(gmsh_curve_tag+1)
                            gmsh_curve_tag+=2
                        if j == 1 and i_1 == 2:
                            gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), (gmsh_surface_tag-3)], gmsh_curve_tag+1)
                            gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                            bottom.append(gmsh_curve_tag+1)
                            gmsh_curve_tag+=2
                        if j > 0 and i_1 > 2:    
                            gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-1), -(gmsh_surface_tag-2), (gmsh_surface_tag-4)], gmsh_curve_tag+1)
                            gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                            bottom.append(gmsh_curve_tag+1)
                            gmsh_curve_tag+=2                    

                    elif(jump_checker):
                        gmsh.model.occ.addLine(c-1, a ,gmsh_surface_tag)
                        jump_checker_true[j].append(gmsh_surface_tag)
                        gmsh_surface_tag+=1

                        gmsh.model.occ.addCurveLoop([-(gmsh_surface_tag-3), gmsh_surface_tag-1, -(gmsh_surface_tag-2)], gmsh_curve_tag+1)
                        gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                        bottom.append(gmsh_curve_tag+1)
                        gmsh_curve_tag+=2
                    
                        jump_checker = False

        # # Flächen für Bottom
        for j in range(resolution-2): 
            for i in range(len(anders_kreis[j])):
                if j == 0 and i== 0: 
                    # print("i0 bottom "+str(i_0_arr_btm))
                    # print("circ bottom "+str(anders_kreis_btm))
                    # print("j > 3 "+str(j_groesser_3))
                    # print("i_0 arr 2 "+str(anders_ecke))
                    gmsh.model.occ.addCurveLoop([-i_0_arr_2[0],-i_0_arr_btm[0],anders_ecke[0][-1] ], gmsh_curve_tag+1)
                    gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                    bottom.append(gmsh_curve_tag+1)
                    gmsh_curve_tag+=2

                for tri in range(len(anders_ecke[j+1])): # anders_ecke durchiterieren und gucken, was passt. Kein Bock mehr. 
                    try: 
                        gmsh.model.occ.addCurveLoop([anders_kreis_btm[j][i],-jump_checker_true[j+1][i] ,anders_ecke[j+1][tri] ], gmsh_curve_tag+1)
                        gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                        bottom.append(gmsh_curve_tag+1)
                        gmsh_curve_tag+=2
                    except:
                        tri+=1

            gmsh.model.occ.addCurveLoop([-anders_ecke[j+1][0], -anders_kreis_btm[j+1][0], i_0_arr_2[j+1]], gmsh_curve_tag+1)
            gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
            bottom.append(gmsh_curve_tag+1)
            gmsh_curve_tag+=2     
            if j < 3: 
                gmsh.model.occ.addCurveLoop([i_0_arr_btm[j], -j_ungleich_0[j], anders_ecke[j+1][-1] ], gmsh_curve_tag+1)
                gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                bottom.append(gmsh_curve_tag+1)
                gmsh_curve_tag+=2      
            else: 
                gmsh.model.occ.addCurveLoop([anders_kreis_btm[j][-1], -j_groesser_3[j-3], anders_ecke[j+1][-1]], gmsh_curve_tag+1)
                gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                bottom.append(gmsh_curve_tag+1)
                gmsh_curve_tag+=2

        schraeg = []
        gerade = []

        # Linien und Flächen Wände: 
        for j in range(0, resolution-1):            # j ist immer der innere Ring
            for i_1 in range(crc_geo[j+1]):             # Triangles pro Ring erzeugen
                c = von[j+1] + i_1                 # aktueller Kreis oben, Punkt 1
                h = (von[j+1]+crc_geo[j+1]*2)-1-crc_geo[j+1]    # Maximalwert des Kreises oben
                if j == resolution-2:
                    if i_1 ==0:
                        e = c+len(vertices_btm)             # aktueller Kreis unten, Punkt 1
                        f = e + 1                           # aktueller Kreis unten, Punkt 2
                        g = h+len(vertices_btm)
                        # erste und letzte gerade wand
                        gmsh.model.occ.addLine(h, g ,gmsh_surface_tag)
                        gerade.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1
                        gmsh.model.occ.addLine(e, c ,gmsh_surface_tag)
                        gerade.append(gmsh_surface_tag)
                        gmsh_surface_tag+=1
                    else: 
                        e = c+len(vertices_btm)             # aktueller Kreis unten, Punkt 1
                        f = e + 1                           # aktueller Kreis unten, Punkt 2
                        g = h+len(vertices_btm)
                        if i_1 != crc_geo[j+1]-1: 
                            gmsh.model.occ.addLine(c, f-1 ,gmsh_surface_tag)
                            gerade.append(gmsh_surface_tag)
                            gmsh_surface_tag+=1

        for i_1 in range(len(anders_kreis[-2])):
            try: 
                gmsh.model.occ.addCurveLoop([-anders_kreis[-2][i_1], gerade[i_1+1],anders_kreis_btm[-2][i_1], -gerade[i_1+2]], gmsh_curve_tag+1)
                gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
                walls.append(gmsh_curve_tag+1)
                gmsh_curve_tag+=2
            except: 
                continue

        gmsh.model.occ.addCurveLoop([-anders_kreis[-2][0], -gerade[1],anders_kreis_btm[-2][0], -gerade[2]], gmsh_curve_tag+1)
        gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
        walls.append(gmsh_curve_tag+1)
        gmsh_curve_tag+=2

        gmsh.model.occ.addCurveLoop([-anders_kreis[-2][-1], gerade[-1],anders_kreis_btm[-2][-1], -gerade[0]], gmsh_curve_tag+1)
        gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
        walls.append(gmsh_curve_tag+1)
        gmsh_curve_tag+=2

        gmsh.model.occ.addCurveLoop([-i_0_arr[-1], gerade[0], i_0_arr_btm[-1], gerade[1]], gmsh_curve_tag+1)
        gmsh.model.occ.addPlaneSurface([-kvar*gmsh_curve_tag+1], gmsh_curve_tag+1)
        walls.append(gmsh_curve_tag+1)
        gmsh_curve_tag+=2

        # # Erstelle das Volumen
        surface_array = up+walls+bottom
        # gmsh.model.occ.synchronize()
        # flchen_vorher = gmsh.model.getEntities(0)
        # gmsh.model.occ.removeAllDuplicates()
        # gmsh.model.occ.synchronize()
        # flchen_nachher = gmsh.model.getEntities(0)
        # print("vor"+str(len(flchen_vorher)))
        # print("nach"+str(len(flchen_nachher)))
        # print("vor"+str(flchen_vorher))
        # print("nach"+str(flchen_nachher))

        tag_to_remove = 3594
        surface_array = [s for s in surface_array if s != tag_to_remove]

        tag_to_remove = 7392
        surface_array = [s for s in surface_array if s != tag_to_remove]

        gmsh.model.occ.addSurfaceLoop(surface_array, volume_tag)
        # gmsh.model.occ.addVolume([volume_tag], volume_tag)

        # # # Synchronisiere die CAD-Entitäten mit dem Gmsh-Modell
        gmsh.model.occ.synchronize()

        gmsh.model.add_physical_group(2, walls, 1)
        gmsh.model.setPhysicalName(2, 1, "Walls")

        gmsh.model.add_physical_group(2, up, 2)
        gmsh.model.setPhysicalName(2, 2, "Up")

        gmsh.model.add_physical_group(2, bottom, 3)
        gmsh.model.setPhysicalName(2, 3, "Bottom")

        gmsh.model.add_physical_group(3, [1], 4)
        gmsh.model.setPhysicalName(3, 4, "Volume")

        # Synchronisiere die CAD-Entitäten mit dem Gmsh-Modell
        # gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        # gmsh.option.setNumber("Mesh.Algorithm",7)  # Try different algorithms (1 to 7)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1) # Muss 1 sein, wenn gmsh eigene Dreiecke baut, gibt es Richtungsfehler!
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)

        # Generiere das 3D-Mesh
        gmsh.model.mesh.generate(3)
        if order > 1: 
            gmsh.model.mesh.setOrder(order)
            gmsh.model.mesh.optimize("HighOrder")
    
        # Wenn du die GUI verwenden möchtest, um das Mesh zu visualisieren
        if '-nopopup' not in sys.argv and gmsh_gui == True:
            gmsh.fltk.run()

        # Speichere das Mesh in einer Datei
        if msh == True: 
            gmsh.write(str(name)+".msh")
        if stl == True: 
            gmsh.write(str(name)+".stl")
        if geo == True: 
            gmsh.write(str(name)+".step")
        gmsh.finalize()  

class collection: 
    def __init__(self): 
        pass
    def meta_1d(): # 2002 Geometrien (bisher)
        pi = np.pi
        k = 10*20
        #meta_collection = np.array([[shape] for _ in range(100000)])
        meta_collection = []

        var = 0
        for i in range(0, 10):
            zeiger_theta = np.linspace(0, (i+1)*2*pi, k)
            for i_2 in range(i, 10):        
                zeiger_theta_2 = np.linspace(0, (i_2+1)*2*pi, k) 
                for i_3 in range(i_2, 10): 
                    zeiger_theta_3 = np.linspace(0, (i_3+1)*2*pi, k)
                    for i_4 in range(i_3, 10): 
                        zeiger_theta_4 = np.linspace(0, (i_4+1)*2*pi, k)
                        for i_5 in range(i_4, 10):
                            zeiger_theta_5 = np.linspace(0, (i_5+1)*2*pi, k)
                            meta_collection.append(((i+1)*np.cos(zeiger_theta)
                                                    +i_2*np.cos(zeiger_theta_2)
                                                    +i_3*np.cos(zeiger_theta_3)
                                                    +i_4*np.cos(zeiger_theta_4)
                                                    +i_5*np.cos(zeiger_theta_5)))
                            var+=1 

        return meta_collection

    def meta_rec(self, **kwargs): 
        samplerate = 44100
        duration = 1
        name = kwargs.get('saveas', "default_rec")
        msh = mesh()

        def start_recording():
            global audio_data
            # Diese Funktion wird beim Klick auf den Button aufgerufen
            print("Aufnahme gestartet...")
            time.sleep(0.5)
            audio_data = sd.rec(
                int(duration * samplerate), 
                samplerate=samplerate, 
                channels=1
                )
            sd.wait()

            audio_data = np.ravel(audio_data) 
            root.after(0, cleanup)

        def cleanup():
            root.quit()
            root.destroy()
            screen_width, screen_height = pyautogui.size()  # Bildschirmgröße ermitteln
            x = screen_width/2  # Position des Schließen-Buttons (X-Position)
            y = screen_height/2  # Position des Schließen-Buttons (Y-Position)

            # Maus zum Schließen-Button bewegen und klicken
            pyautogui.click(x, y)

        # Funktion zur Aufnahmesteuerung in einem separaten Thread
        def record_in_thread():
            threading.Thread(target=start_recording).start()

        # GUI mit tkinter
        root = tk.Tk()
        root.title("O")

        # Fenstergröße festlegen
        root.geometry("300x150")

        header_label = tk.Label(root, text="Singe einen Vokal", font=("Arial", 16))
        header_label.pack(pady=20)
        # Button zur Aufnahme
        record_button = tk.Button(root, text="O", command=record_in_thread, padx=10, pady=5, fg='red')
        record_button.pack(pady=20)

        # Haupt-Event-Schleife der GUI
        root.mainloop()

        fft = np.fft.fft(audio_data)[50:4100]
        peaks, _ = find_peaks(abs(fft), 100)
        one_period_length = int(44100/(peaks[0]+50))
        if one_period_length%2 != 0: 
            one_period_length-=1

        max_one_period = np.argmax(audio_data[5000:5000+one_period_length])
        audio_data_one_period = audio_data[5000+max_one_period:5000+max_one_period+one_period_length]

        #audio_data_one_period = np.subtract(np.divide(audio_data_one_period, np.max(audio_data_one_period)), 0.1)[::10]
        stepz = int(len(audio_data_one_period)/20)
        audio_data_one_period = np.divide(audio_data_one_period, np.max(audio_data_one_period))[::stepz]

        #audio_data_one_period = np.concatenate(([audio_data_one_period[0]], audio_data_one_period))
        #audio_data_one_period = np.concatenate((audio_data_one_period, [audio_data_one_period[len(audio_data_one_period)-1]]))

        res =5
        # FENICS Mesh
        [x_geo, y_geo, z_geo, crc_geo, phi_arr_geo, x_btm, y_btm, z_btm, crc_btm, phi_arr_btm, hoehe
        ] = msh.nodes_2D_rec(
                plot=True, 
                theta=audio_data_one_period, 
                resolution=res, 
                #resolution=10,
                period=2,
                bottoms=0,
                bottom_position=2,
                space = 0.01, 
                hoehe=100
                )
        
        x_btm = np.array(x_btm)
        y_btm = np.array(y_btm)
        z_btm = np.array(z_btm)

        msh.material4(x_geo, 
                      y_geo, 
                      z_geo, 
                      crc_geo, 
                      phi_arr_geo, 
                      x_btm, 
                      y_btm, 
                      z_btm, 
                      crc_btm, 
                      phi_arr_btm, 
                      saveas=name,
                      counterpart=True, 
                      lc=0.025,
                      lc_geo=0.025,
                      resolution=res,
                      rec = True,
                      mesh_order=2, 
                      generate_stl = False,
                      )

        ######################### STL 
        print("geomin" +str(np.min(z_geo)))
        [x_geo, y_geo, z_geo, crc_geo, phi_arr_geo, x_btm, y_btm, z_btm, crc_btm, phi_arr_btm, hoehe
        ] = msh.nodes_2D_rec(
                plot=True, 
                theta=audio_data_one_period, 
                resolution=res, 
                #resolution=10,
                period=2,
                bottoms=0,
                #bottom_position=np.min(z_geo)*2.01, funktioniert irgendwie nicht
                bottom_position=-0.11,
                space = 0.01, 
                hoehe=100
                )
        
        x_btm = np.array(x_btm)
        y_btm = np.array(y_btm)
        z_btm = np.array(z_btm)

        msh.material4(x_geo, 
                      y_geo, 
                      z_geo, 
                      crc_geo, 
                      phi_arr_geo, 
                      x_btm, 
                      y_btm, 
                      z_btm, 
                      crc_btm, 
                      phi_arr_btm, 
                      saveas=name+str("_for_print"),
                      counterpart=False, 
                      lc=0.025,
                      lc_geo=0.025,
                      resolution=res,
                      rec = True,
                      mesh_order=2, 
                      generate_stl = True,
                      )

        return audio_data_one_period
    
    def meta_random(self, **kwargs): 
        name = kwargs.get("saveas", "default_random_geo")

        msh = mesh()
        res = kwargs.get("resolution", 5)

        [x_geo, 
        y_geo, 
        z_geo, 
        crc_geo, 
        phi_arr_geo, 
        x_btm, 
        y_btm, 
        z_btm, 
        crc_btm, 
        phi_arr_btm, 
        hoehe] = msh.nodes_2D(
            plot=False, 
            period=0, 
            resolution=res, 
            bottom_position=3.2,
            bottoms=0, 
            space = 0.01,
            durchmesser=102,
            hoehe=50)
        for i in range(len(z_geo)): 
           z_geo[i] = random.uniform(-0.1, 0.1)

        msh.plotten3d_multi(x_geo, y_geo, z_geo, x_btm, y_btm, z_btm)

        msh.material4(
            x_geo, 
            y_geo, 
            z_geo, 
            crc_geo, 
            phi_arr_geo, 
            x_btm, 
            y_btm, 
            z_btm, 
            crc_btm, 
            phi_arr_btm, 
            lc = 0.01, 
            lc_geo = 0.01, 
            saveas=name,
            counterpart=True, 
            resolution=res, 
            gmsh_gui=False)

        print("here is the print: ")

        [x_geo, 
        y_geo, 
        z_geo_2, 
        crc_geo, 
        phi_arr_geo, 
        x_btm, 
        y_btm, 
        z_btm, 
        crc_btm, 
        phi_arr_btm, 
        hoehe] = msh.nodes_2D(
            plot=False, 
            period=0, 
            resolution=res, 
            bottom_position=-0.15,
            bottoms=0, 
            space = 0.01,
            durchmesser=102,
            hoehe=50)     

        z_geo_2 = z_geo

        msh.plotten3d_multi(x_geo, y_geo, z_geo_2, x_btm, y_btm, z_btm)

        msh.material4(
            x_geo, 
            y_geo, 
            z_geo_2, 
            crc_geo, 
            phi_arr_geo, 
            x_btm, 
            y_btm, 
            z_btm, 
            crc_btm, 
            phi_arr_btm, 
            lc = 0.01, 
            lc_geo = 0.01, 
            saveas=name+"_for_print",
            counterpart=False, 
            generate_stl = True, 
            generate_msh = False, 
            resolution=res, 
            gmsh_gui=False)

class fem_postproc: 
    def __init__(self): 
        pass

    def plot_2d(self, r, files, name, f_axis, f_low, f_high, **kwargs):

        save_r = kwargs.get("save_r", False)
        plot_r = kwargs.get("plot_r", True)
        short_legend = kwargs.get("short_legend", 30)
        is_r = kwargs.get("is_r", True)

        fig = go.Figure()
        for i in range(len(files)): 
            trace = go.Scatter(x=f_axis, y=r[i], name=files[i][-short_legend:])
            fig.add_trace(trace)
        if is_r: 
            fig.update_yaxes(range=[0, 1.1])
        fig.update_xaxes(range=[f_low, f_high])

        fig.update_layout(
            title="\""+str(name)+"\" - <br> Finite Element Simulation",
            xaxis_title="Frequency [Hz]",
            yaxis_title="Reflection Coefficient",
            #legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=20,
                color="RebeccaPurple"
            )
            )

        if save_r == True: 
            pio.write_html(fig, './'+name+'.html')
            pio.write_image(fig, './'+name+'.png', width=1080, height=720, scale=10)

        if plot_r == True: 
            fig.show() 

    # xyz wird wichtig, sobald vektorielle Helmholtzgleichung gelöst wird
    def xyz_vector(self, all_values, f_axis):
        bis = len(all_values[0][0])

        p_x_re = [[] for _ in range(len(f_axis))]
        p_y_re = [[] for _ in range(len(f_axis))]
        p_z_re = [[] for _ in range(len(f_axis))]
        p_mag_re = [[] for _ in range(len(f_axis))]

        p_x_im = [[] for _ in range(len(f_axis))]
        p_y_im = [[] for _ in range(len(f_axis))]
        p_z_im = [[] for _ in range(len(f_axis))]
        p_mag_im = [[] for _ in range(len(f_axis))]

        for i in range(len(f_axis)):
            for j in range(bis):
                if j % 3 == 0: 
                    p_x_re[i].append(all_values[i][0][j])
                    p_x_im[i].append(all_values[i][1][j])
                if j % 3 == 1: 
                    p_y_re[i].append(all_values[i][0][j])
                    p_y_im[i].append(all_values[i][1][j])
                if j % 3 == 2: 
                    p_z_re[i].append(all_values[i][0][j])
                    p_z_im[i].append(all_values[i][1][j])

            p_mag_re[i].append(np.sqrt(np.array(np.square(p_x_re[i])) + np.square(np.array(p_y_re[i])) + np.square(np.array(p_z_re[i]))))
            p_mag_im[i].append(np.sqrt(np.array(np.square(p_x_im[i])) + np.square(np.array(p_y_im[i])) + np.square(np.array(p_z_im[i]))))

        return p_x_re, p_y_re, p_z_re, p_x_im, p_y_im, p_z_im, p_mag_re, p_mag_im
    def xyz_vector_eigen(self, all_values, f_axis): 
        bis = len(all_values[0])
        p_x_re = [[] for _ in range(len(f_axis))]
        p_y_re = [[] for _ in range(len(f_axis))]
        p_z_re = [[] for _ in range(len(f_axis))]
        p_mag_re = [[] for _ in range(len(f_axis))]

        for i in range(len(f_axis)):
            for j in range(bis):
                if j % 3 == 0: 
                    p_x_re[i].append(all_values[i][j])
                if j % 3 == 1: 
                    p_y_re[i].append(all_values[i][j])
                if j % 3 == 2: 
                    p_z_re[i].append(all_values[i][j])

            p_mag_re[i].append(np.sqrt(np.array(np.square(p_x_re[i])) + np.square(np.array(p_y_re[i])) + np.square(np.array(p_z_re[i]))))

        return p_x_re, p_y_re, p_z_re, p_mag_re
    
    def reflection_coefficient(self, mags, f_axis, files, name, f_low, f_high, **kwargs): 
        fig = go.Figure()
        all_mean_mag = [[] for _ in range(len(mags))]

        save_r = kwargs.get("save_r", True)
        plot_r = kwargs.get("plot_r", True)
        von = kwargs.get("von", 30000)
        bis = kwargs.get("bis", 100000)
        axis_int = kwargs.get("axis_int", False)

        if axis_int == True: 
            f_axis = np.arange(len(f_axis))
            f_low = 0
            f_high = len(f_axis)

        traces = []
        if len(files)> 1: 
            r = [[] for _ in range(len(mags))]
            for j in range(len(mags)): 
                for i in range(len(f_axis)):
                    all_mean_mag[j].append(np.max(mags[j][i][von:bis]))

                    r[j].append(
                        (1-np.min(mags[j][i][von:bis])/np.max(mags[j][i][von:bis]))/
                        (1+np.min(mags[j][i][von:bis])/np.max(mags[j][i][von:bis]))
                        )
                    
                #traces.append(go.Scatter(x=f_axis, y=all_mean_mag[j], name=files[j]))
                traces.append(go.Scatter(x=f_axis, y=r[j], name=files[j][-30:]))
                fig.add_trace(traces[j])
        else:
            r=[]
            for i in range(len(f_axis)):
                # all_mean_mag.append(np.max(mags[i][von:bis]))

                r.append(
                    (1-np.min(mags[i][von:bis])/np.max(mags[i][von:bis]))/
                    (1+np.min(mags[i][von:bis])/np.max(mags[i][von:bis]))
                    )

            traces.append(go.Scatter(x=f_axis, y=r, name=files[0]))
            fig.add_trace(traces[0])

        fig.update_yaxes(range=[0, 1.1])
        fig.update_xaxes(range=[f_low, f_high])

        fig.update_layout(
            title="\""+str(name)+"\" - <br> Finite Element Simulation",
            xaxis_title="Frequency [Hz]",
            yaxis_title="Reflection Coefficient",
            #legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=20,
                color="RebeccaPurple"
            )
            )
        
        import kaleido
        import plotly.io as pio

        if save_r == True: 
            pio.write_html(fig, './'+name+'.html')
            pio.write_image(fig, './'+name+'.png', width=1080, height=720, scale=10)

        if plot_r == True: 
            fig.show()
        
        return r

    def reflection_coefficient_correct_shift(self, mags, f_axis, files, **kwargs): 
        fig = go.Figure()
        all_mean_mag = [[] for _ in range(len(mags))]
        r = [[] for _ in range(len(mags))]

        save_r = kwargs.get("save_r", True)
        plot_r = kwargs.get("plot_r", True)
        von = kwargs.get("von", 30000)
        bis = kwargs.get("bis", len(mags[0][0])-von)

        traces = []

        for j in range(len(mags)): 
            for i in range(len(f_axis)):
                all_mean_mag[j].append(np.max(mags[j][i][von:bis]))

                mini = np.min(mags[j][i][von:bis])
                maxi = np.max(mags[j][i][von:bis])

                if maxi > 2:
                    mini = 0
                    maxi = 2

                r[j].append(
                    (1-(mini/maxi))/
                    (1+(mini/maxi))
                    )
                
            #traces.append(go.Scatter(x=f_axis, y=all_mean_mag[j], name=files[j]))
            traces.append(go.Scatter(x=f_axis, y=r[j], name=files[j][-10]))
            fig.add_trace(traces[j])

        # traces.append(go.Scatter(x=f_axis, y=np.subtract(all_mean_mag[2], all_mean_mag[1]), name="schauen"))
        # fig.add_trace(traces[j+1])

        fig.update_yaxes(range=[0, 1.1], title="Reflection Coefficient")
        fig.update_xaxes(title="Frequency [Hz]")

        fig.update_layout(
            title="Reflection Coefficient"
        )

        if save_r == True: 
            fig.write_html("./big_faxis.html")

        if plot_r == True: 
            fig.show()
        
        return r

    def load_solutions(self, **kwargs): 
        path = kwargs.get("path", "./all_values_npy/R1_revised/Solution/")
        files=glob.glob(path+"*npy")
        files.sort()

        all_meas = [[]for _ in range(len(files))]

        mags = []

        for i in range(len(files)): 
            all_meas[i].append(np.load(files[i]))
            print(files[i])

            mags.append(np.sqrt(np.square(np.load(files[i])[:, 0]) + np.square(np.load(files[i])[:, 1])))

        all_meas = np.squeeze(all_meas)
        mags = np.squeeze(mags)

        return all_meas, mags, files

    def load_one_solution(self, **kwargs): 
        path = kwargs.get("path", "./all_values_npy/R1_revised/Solution/")
        file_index = kwargs.get("file_index", 0)
        files=glob.glob(path+"*npy")
        files.sort()

        mags = []

        #for i in range(len(files)): 
        print(files[file_index])

        mags.append(np.sqrt(np.square(np.load(files[file_index])[:, 0]) + np.square(np.load(files[file_index])[:, 1])))

        mags = np.squeeze(mags)

        return mags, files

    def merger(self, path, **kwargs):
        order = kwargs.get("order", None)

        files = glob.glob(path+"*npy")
        files.sort()

        files_sorted = []
        if order != None and order != 0: 
            for i in order: 
                files_sorted.append(files[i])
    
        all_values = []

        for i in range(len(files)):
            if order == None: 
                print(str(i)+" "+str(files[i]))
                data = np.load(files[i])
                for j in range(len(data)): 
                    all_values.append(data[j])
            else: 
                print(str(i)+" "+str(files_sorted[i]))
                data = np.load(files_sorted[i])
                for j in range(len(data)): 
                    all_values.append(data[j])


        all_values = np.array(all_values)
        np.save(path+"_merged.npy", all_values)

        return all_values, files

    def plot_pressure(self, all_meas, files, mags, f_axis, **kwargs): 
        fig_real = go.Figure()
        fig_imag = go.Figure()
        fig_mag = go.Figure()

        plot_real = kwargs.get("plot_real", True)
        plot_imag = kwargs.get("plot_imag", True)
        plot_mag = kwargs.get("plot_mag", True)

        freq = kwargs.get("freq", 150)

        traces_real = []
        traces_imag=[]
        traces_mag=[]

        if len(files) > 1: 
            for i in range(len(files)): 
                traces_real.append(go.Scatter(x=np.arange(len(all_meas[i, freq, 0])), y=all_meas[i, freq, 0], name=files[i][-30:]+" real"))
                traces_imag.append(go.Scatter(x=np.arange(len(all_meas[i, freq, 0])), y=all_meas[i, freq, 1], name=files[i][-30:]+" imag"))
                traces_mag.append(go.Scatter(x=np.arange(len(mags[i][freq])), y=mags[i][freq], name=files[i][-30:]+" mag"))

                fig_real.add_trace(traces_real[i])
                fig_imag.add_trace(traces_imag[i])
                fig_mag.add_trace(traces_mag[i])
        else: 
                traces_real.append(go.Scatter(x=np.arange(len(all_meas[freq, 0])), y=all_meas[freq, 0], name=files[0][-30:]+" real"))
                traces_imag.append(go.Scatter(x=np.arange(len(all_meas[freq, 0])), y=all_meas[freq, 1], name=files[0][-30:]+" imag"))
                traces_mag.append(go.Scatter(x=np.arange(len(mags[freq])), y=mags[freq], name=files[0][-30:]+" mag"))

                fig_real.add_trace(traces_real[0])
                fig_imag.add_trace(traces_imag[0])
                fig_mag.add_trace(traces_mag[0])


        fig_real.update_layout(
            title=str(f_axis[freq])+" real"
        )

        fig_imag.update_layout(
            title=str(f_axis[freq])+ " imag"
        )

        fig_mag.update_layout(
            title=str(f_axis[freq])+ " mags"
        )

        fig_mag.update_yaxes(range=[0, 5])

        if plot_real == True: 
            fig_real.show()
        if plot_imag == True: 
            fig_imag.show()
        if plot_mag == True: 
            fig_mag.show()

    def plot_modes(self, nodes_path, mags, f_axis, **kwargs): 
        nodes = np.load(nodes_path)
        # dofs_geo = np.load("./all_values_npy/R1_revised/velo_dofs/r1_2nd_order_revised_robin_1_revised_mesh_velo_dofs.npy")
        # nodes_pla_re = []

        # nodes_pla_re.append(nodes[:, 0])
        # nodes_pla_re.append(nodes[:, 1])
        # nodes_pla_re.append(nodes[:, 2])

        freq = kwargs.get("freq", 140)

        von = kwargs.get("von", 0)
        bis = kwargs.get("bis", len(mags[0][0]))
        meas = kwargs.get("meas", 0)

        fig= go.Figure(data=[go.Scatter3d(x=nodes[:, 0][von:bis], 
                                        y=nodes[:, 1][von:bis], 
                                        z=nodes[:, 2][von:bis],
                                        mode='markers',
                                        marker=dict(size=1,
                                                    color=mags[meas][freq][von:bis],
                                                    #color=all_meas[2, freq, 1],
                                                    colorscale = "viridis", 
                                                    )
                                        )])
        fig.update_layout(
            title=str(f_axis[freq])+" Hz",
            title_font_size=20, 
            title_x=0.5,
                xaxis=dict(scaleanchor="y"),  # Erzwingt gleiche Skalierung
                yaxis=dict(scaleanchor="x"),   # Alternativ könnte nur eine Angabe reichen
                scene=dict(
                    xaxis_title="X [m]",
                    yaxis_title="Y [m]",
                    zaxis_title="Z [m]",
                ),
            margin=dict(l=0, r=0, t=50, b=0), 
            scene_camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.6)  # Kamera weiter zurücksetzen für weniger Zoom
            )
        )

        fig.show()

    def find_indices(self, nodes_path): 
        nodes = np.load(nodes_path)

        index_arr = np.arange(len(nodes))

        #fig = px.scatter_3d(x=nodes[:, 0][von:bis],y=nodes[:, 1][von:bis],z=nodes[:, 2][von:bis], hover_data=[index_arr[von:bis]])
        fig = px.scatter_3d(x=nodes[:, 0],y=nodes[:, 1],z=nodes[:, 2], hover_data=[index_arr])
        fig.update_traces(marker=dict(size=1))
        fig.show()

    def tube_animation(self, f_axis, mag, msh_nodes_path, name, **kwargs): 
        #msh_nodes = np.load("./all_values_npy/r_1_long_tube_finer_msh_nodes.npy")
        msh_nodes = np.load(msh_nodes_path+name+".npy")

        unit = kwargs.get("Unit", "Pressure")

        x = msh_nodes[:, 0]
        y = msh_nodes[:, 1]
        z = msh_nodes[:, 2]

        if not os.path.exists("./pngtemp"):
            os.makedirs("./pngtemp")

        def parallel(von, bis): 
            for i in range(von, bis):
                fig = go.Figure(data=[go.Scatter3d(x=x, 
                                                y=y, 
                                                z=z,
                                                mode='markers',
                                                marker=dict(size=8,
                                                            #color=mag[i][0],
                                                            color=mag[i],
                                                            colorscale = "viridis", 
                                                            colorbar=dict(title=unit),
                                                            )
                                                )])
                fig.update_layout(
                    title=str(f_axis[i])+" Hz",
                    title_font_size=20, 
                    title_x=0.5,
                        scene=dict(
                            xaxis_title="X [m]",
                            yaxis_title="Y [m]",
                            zaxis_title="Z [m]",
                        ),
                    margin=dict(l=0, r=0, t=50, b=0), 
                    scene_camera=dict(
                        eye=dict(x=1.6, y=1.6, z=1.6)  # Kamera weiter zurücksetzen für weniger Zoom
                    )
                )
                fig.write_image("pngtemp/"+str(f_axis[i])+".png")

        von = 0
        von_2 = int(len(f_axis)/4)
        von_3 = int(len(f_axis)/4)*2
        von_4 = int(len(f_axis)/4)*3

        bis = von_2-1
        bis_2 = von_3-1
        bis_3 = von_4-1
        bis_4 = len(f_axis)-1


        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit each function to run concurrently
            future1 = executor.submit(parallel, von, bis)
            future2 = executor.submit(parallel, von_2, bis_2)
            future3 = executor.submit(parallel, von_3, bis_3)
            future4 = executor.submit(parallel, von_4, bis_4)

            # Wait for both futures to complete
            concurrent.futures.wait([future1, future2, future3, future4])

        # Pfad zu deinem Ordner mit den PNG-Dateien
        image_folder = './pngtemp'
        video_name = str(name)+'.mp4'

        # Lies alle Bilder im Ordner

        images = []

        for i in range(len(f_axis)): 
            images.append(str(f_axis[i])+".png")

        #images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        #print(images)
        #images.sort()  # Sortieren nach Namen (falls nötig)

        # Setze die Video-Einstellungen
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec für mp4-Dateien
        video = cv2.VideoWriter(video_name, fourcc, 15, (width, height))  # 30 FPS

        # Füge die Bilder zum Video hinzu
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        # Beende das Video
        cv2.destroyAllWindows()
        video.release()

        # Verzeichnis und alle Inhalte löschen
        shutil.rmtree(image_folder)
