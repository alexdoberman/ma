# -*- coding: utf-8 -*-
import numpy as np

def get_delays_line_geom(arraygeometry, avgDOA):
    delays = []
    refX   = 0
    sspeed = 343000.0         # Sound speed (mm/s)
    chanN  = len(arraygeometry)
    for chX in range(chanN):
        dist = abs( arraygeometry[chX][0] - arraygeometry[refX][0] )        # This is from original codes but it is probably wrong if you are not careful with avgDOA
        timedelay = dist * np.cos( avgDOA * np.pi / 180.) / sspeed
        delays.append(timedelay)
    return np.array(delays)

def get_delays_3d_geom(arraygeometry, avgTheta, avgPhi):
    delays = []
    refX   = 0
    sspeed = 343000.0         # Sound speed (mm/s)
    chanN, dim  = arraygeometry.shape
    avgTheta = avgTheta* np.pi / 180.
    avgPhi = avgPhi* np.pi / 180.
    
    for chX in range(chanN):
        if refX == chX:
            delays.append(0.0)
        else:
            '''
            v1 = - np.array([np.sin(avgTheta)*np.cos(avgPhi), np.sin(avgTheta)*np.sin(avgPhi), np.cos(avgTheta)])     # Unit vector of the sound arrival direction
            v2 = np.array(arraygeometry[refX]) - np.array(arraygeometry[chX])     # Vector in the direction from reference channel to current channel
            v1_l = 1
            v2_l = np.sqrt(sum(np.multiply(v2,v2)))                                               # Distance between two channels
            dist = v2_l
            v1_dot_v2 = sum(np.multiply(v1,v2))
            cos_avgDOAs = v1_dot_v2/(v1_l*v2_l) # Cosine of the angle between the two vectors
            timedelay = - np.sign(v1_dot_v2) * dist * cos_avgDOAs / sspeed		# The minus sign is needed because that is how other modules in BTK works! Confusing? Yes!
            delays.append(timedelay)
            '''
            v1 = np.array([np.sin(avgTheta)*np.cos(avgPhi), np.sin(avgTheta)*np.sin(avgPhi), np.cos(avgTheta)])     # Unit vector of the sound arrival direction
            v2 = np.array(arraygeometry[refX]) - np.array(arraygeometry[chX])     # Vector in the direction from reference channel to current channel
            v1_l = 1
            v2_l = np.sqrt(sum(np.multiply(v2,v2)))                                               # Distance between two channels
            dist = v2_l
            v1_dot_v2 = sum(np.multiply(v1,v2))
            cos_avgDOAs = v1_dot_v2/(v1_l*v2_l) # Cosine of the angle between the two vectors
            timedelay = dist * cos_avgDOAs / sspeed		# The minus sign is needed because that is how other modules in BTK works! Confusing? Yes!
            delays.append(timedelay)
    return np.array(delays)



def get_11_geometry():
    micPositions = []

    micPositions.append([ 0.0,0,0])
    micPositions.append([ 1*35.0,0,0])
    micPositions.append([ 2*35.0,0,0])
    micPositions.append([ 3*35.0,0,0])
    micPositions.append([ 4*35.0,0,0])
    micPositions.append([ 5*35.0,0,0])
    micPositions.append([ 6*35.0,0,0])
    micPositions.append([ 7*35.0,0,0])
    micPositions.append([ 8*35.0,0,0])
    micPositions.append([ 9*35.0,0,0])
    micPositions.append([ 10*35.0,0,0])

    micPositions = np.array(micPositions)
    return micPositions


def get_66_geometry():

    Hor_mic_count  = 11
    Vert_mic_count = 6
    dHor           = 35
    dVert          = 50

    micPositions = []
    for v in range(0,Vert_mic_count,1):
        for h in range(0,Hor_mic_count,1):
            x =  h * dHor
            y =  v * dVert
            z = 0

            micPositions.append([x,y,z])

    micPositions = np.array(micPositions)
    return micPositions

def recalc_angle(a, b):
    # a = angle_hor, b = angle_vert

    a = a*(np.pi/180.)
    b = b*(np.pi/180.)

    avgPhi = np.arctan(1.0/(np.tan(a)*np.tan(np.pi/2 - b)))
    avgTheta = np.arctan(np.tan(a)/np.cos(avgPhi))
    avgTheta1 = np.arctan(1.0/(np.sin(avgPhi)*np.tan(np.pi/2 - b)))

    avgPhi = avgPhi*(180./np.pi)
    avgTheta = avgTheta*(180./np.pi)
    return avgPhi, avgTheta
    