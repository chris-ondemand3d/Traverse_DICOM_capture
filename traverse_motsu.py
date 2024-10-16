import sys
import numpy
from pathlib import Path
import subprocess, time
from PIL import Image as im

import pymongo
import gdcm
import os, glob, json
from bson import ObjectId
import matplotlib.pyplot as plt

from PySide6.QtGui import QGuiApplication, QWindow
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterType
from PySide6.QtCore import Qt, QObject, QFileInfo, Signal, Slot
from PySide6.QtQuick import QQuickView
from PySide6.QtWidgets import QApplication, QPushButton

from vtkmodules.vtkRenderingCore import vtkActor
# load implementations for rendering and interaction factory cla
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkInteractionStyle
import vtkmodules.util.numpy_support
from vtkmodules.vtkCommonTransforms import vtkTransform
import QVTKRenderWindowInteractor as QVTK
QVTKRenderWindowInteractor = QVTK.QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionWidgets import (
    vtkBoxWidget,
    vtkBoxWidget2,
    vtkBoxRepresentation
)
import ScanDirectory 

import vtk
import nibabel as nib

if QVTK.PyQtImpl == 'PySide6':
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMainWindow

import ScanDirectory

from scipy.ndimage import label
MinConnectedVoxel = 5000


# Extent class with vertex iterator
class Extent:
    def __init__(self, x1, x2, y1, y2, z1, z2):
        self.x1, self.x2 = x1, x2
        self.y1, self.y2 = y1, y2
        self.z1, self.z2 = z1, z2

    def vertices(self):
        for x in (self.x1, self.x2):
            for y in (self.y1, self.y2):
                for z in (self.z1, self.z2):
                    yield (x, y, z)


def find_connected_elements_3d(m, v):
    # Create a binary mask where elements equal to v are True, others are False
    mask = (m == v)
    
    # Use scipy's label function to identify connected components
    labeled_array, num_features = label(mask)
    
    # Get the indices for each labeled region
    indices = [numpy.argwhere(labeled_array == i) for i in range(1, num_features + 1)]
    
    return indices

def calculate_3d_extent(np_image, value):
    """
    Calculate the extent (x1, x2, y1, y2, z1, z2) of a 3D numpy array where elements have a specific value.
    
    Parameters:
    np_image (numpy.ndarray): 3D numpy array
    value: The specific value to search for in the array
    
    Returns:
    tuple: (x1, x2, y1, y2, z1, z2) where (x1, y1, z1) is the minimum extent and (x2, y2, z2) is the maximum extent
    """
    # Find the indices where the value occurs
    connected_indices = find_connected_elements_3d(np_image, value)
    print("num of connected_indices:", len(connected_indices))

    firstTime = True
    for i, group in enumerate(connected_indices):
        #print(value, i, group,group.shape)
        if (group.shape)[0]>=MinConnectedVoxel:
            if (firstTime): 
                indices = numpy.copy(group)
                firstTime = False
            else: indices = numpy.concatenate((indices, group), axis=0)            
    
    if (not firstTime): # Calculate the extents
        x1, x2 = numpy.min(indices[:,0]), numpy.max(indices[:,0])
        y1, y2 = numpy.min(indices[:,1]), numpy.max(indices[:,1])
        z1, z2 = numpy.min(indices[:,2]), numpy.max(indices[:,2])
        return x1, x2, y1, y2, z1, z2
    else:
        return 0,0,0,0,0,0

# parse dicom directory for study info
def parse_dicomfile(dicompath):
    t1 = gdcm.Tag(0x10, 0x20);  # Patient ID
    t2 = gdcm.Tag(0x10, 0x10);  # Patient Name
    t3 = gdcm.Tag(0x08, 0x20);  # Study Date
    t4 = gdcm.Tag(0x20, 0x0d);  # Study Instance UID

    # Iterate over directory
    d = gdcm.Directory();
    nfiles = d.Load(dicompath);
    if (nfiles == 0): sys.exit(1);

    filenames = d.GetFilenames()
    # print("Files ", filenames)

    #  Get rid of any Warning while parsing the DICOM files
    gdcm.Trace.WarningOff()

    # instanciate Scanner:
    sp = gdcm.Scanner.New();
    s = sp.__ref__()
    s.AddTag(t1);
    s.AddTag(t2);
    s.AddTag(t3);
    s.AddTag(t4);

    b = s.Scan(filenames);
    if (not b): sys.exit(1);

    pttv = gdcm.PythonTagToValue(s.GetMapping(filenames[0]))
    pttv.Start()
    # iterate until the end:
    while (not pttv.IsAtEnd()):
        # get current value for tag and associated value:
        # if tag was not found, then it was simply not added to the internal std::map
        # Warning value can be None
        tag = pttv.GetCurrentTag()
        value = pttv.GetCurrentValue()

        if (tag == t1):
        # print ("PatientID->",value)
            patient_id = value
        elif (tag == t2):
        # print ("PatientName->",value)
            patient_name = value        
        elif (tag == t3):
        # print ("Studydate->",value)
            study_date = value      
        elif (tag == t4):
        # print ("StudyInstanceUID->",value)
            study_uid = value

        pttv.Next()                

    return patient_id, patient_name, study_uid, study_date, len(filenames)


class MAINApp(QQuickView):
    
    def __init__(self, exec_param=None, *args, **kwds):
        super().__init__(*args, **kwds)


class fill_json:
    def __init__(self,path):

        # Initialize Window and variables
        self.window = QMainWindow()
        self.window.resize(960,960)
        self.widget = QVTKRenderWindowInteractor(self.window)
        self.window.setCentralWidget(self.widget)
        self.ren = vtk.vtkRenderer()
        self.widget.GetRenderWindow().AddRenderer(self.ren)
        self.ren.SetUseDepthPeeling(True)
        self.ren.UseDepthPeelingForVolumesOn()

        self.volume = vtk.vtkVolume()
        self.screenshotCount = 0
        self.rendermode = 3
        self.renderDir = 3
        self.makeNifti = True
        self.spacing = [.0, .0, .0]
        self.rootpath = path
        
        self.window.move(10,10)
        self.window.show()
        istyle = vtk.vtkInteractorStyleTrackballCamera()
        self.widget._Iren.SetInteractorStyle(istyle)
        self.widget.Initialize()
        self.widget.Start()    
        
        self.outdir = os.path.dirname(os.path.abspath("c:\\dev\Testdata\\out"))
        os.makedirs(self.outdir, exist_ok=True)

        print("Render width, height:", self.widget.width(), self.widget.height())


    def reset(self):
        self.ren.RemoveVolume(self.volume)
        self.volume = None
        

    def capture(self,path):

        with open('render.json') as f:
            self.data = json.load(f)
        self.data["path"] = path

        patient_id, patient_name, study_uid, study_date, nImg = parse_dicomfile(path)
        print("DICOM folder:", path)
        self.data["patient_id"] = patient_id
        self.data["patient_name"] = patient_name
        self.data["study_uid"] = study_uid
        self.data["study_date"] = study_date
    
        self.min_npV, self.max_npV, self.adjThresholds, self.spacing, self.volume = ScanDirectory.load_dicom(path, 1, True) 
        self.ren.AddVolume(self.volume)   

        # Camera Reset and Render
        self.renderDirection(1)        

        # Segmentation  
        # Create input_folder and output_folder -> current folder - segment, output
        input_dir = os.path.join(path,'segment')
        output_dir = os.path.join(path,'output')
 
        # 1,2,3,4,5각각에 대해서, ijk extent -> xyz extent in world coordinate 
        outfile = os.path.join(output_dir,"Dental_0001.nii.gz")
        if os.path.exists(outfile):
            img = nib.load(outfile)            
            npImage = (img.get_fdata()).transpose(2,1,0)
            unique_labels = numpy.unique(npImage)
            extent = []
            self.wcs_extent = []
            xo,yo,zo = self.volume.GetOrigin()
            seg_label=[]
            for i in range(1,6):
                if (i not in unique_labels): # not segmented
                    extent.append([0,0,0,0,0,0])
                    self.wcs_extent.append([.0,.0,.0,.0,.0,.0])
                    seg_label.append(False)
                else:
                    seg_label.append(True)
                    extent.append(calculate_3d_extent(npImage,i))   
                    k1, k2, j1, j2, i1, i2 = extent[i-1]
                    #(i1, j1, k1) to (x1, y1, z1)
                    x1 = i1*self.spacing[0]+xo
                    y1 = j1*self.spacing[1]+yo
                    z1 = k1*self.spacing[2]+zo
                    #(i2, j2, k2) to (x2, y2, z2)
                    x2 = i2*self.spacing[0]+xo
                    y2 = j2*self.spacing[1]+yo
                    z2 = k2*self.spacing[2]+zo                    
                    self.wcs_extent.append([x1,x2,y1,y2,z1,z2])
                    print(extent[i-1], self.wcs_extent[i-1], self.volume.GetBounds())                 

        # Landmark Detection
        # save to "study_uid.txt" in path directory
        start_time=time.time()
        outfile = os.path.join(path,study_uid+".txt")
        lmdetect_command = (
                f".\nnUNetv2_predict -i {input_dir}/ -o {output_dir}/ "
                "-d Dataset111_453CT -tr nnUNetTrainer -p nnUNetPlans "
                "-c 3d_fullres -f 0 -npp 1 -nps 1 -step_size 0.5 -device cuda --disable_tta"
        )
        # subprocess.run(lmdetect_command, shell=True) 
        if os.path.exists(outfile):
            end_time = time.time()
            print("Landmark Detected",end_time-start_time)

        # rendering capture to save png and text file 
        #render mode = 1,2,3
        #render_dir = anterior, right, superior 

        for i in range(1,4):
            self.renderDirection(i)
            for j in range(2,4):
                self.renderMode(j)
                self.captureScreen(study_uid)  

        self.renderDirection(2) # from Left
        self.renderMode(4) # Bone 2
        self.captureScreen(study_uid)
        self.renderMode(5) # soft tissue 2
        self.captureScreen(study_uid)
        self.renderMode(6) # soft tissue 3
        self.captureScreen(study_uid)
        self.renderMode(7) # soft tissue 4
        self.captureScreen(study_uid)

        json_file = os.path.join(self.outdir,'%s.json' % study_uid)
        with open(json_file, 'w') as f:
            json.dump(self.data,f)

    
    def renderDirection(self,direction):
            
        # 1st entering, setup camera   
        fp = numpy.array(self.ren.GetActiveCamera().GetFocalPoint())
        p = numpy.array(self.ren.GetActiveCamera().GetPosition())
        dist = self.ren.GetActiveCamera().GetDistance()
        if direction == 1: # Head to Anterior      
            self.ren.GetActiveCamera().SetPosition(fp[0], fp[1] - dist, fp[2])
            self.ren.GetActiveCamera().SetViewUp(0.0, 0.0, 1.0)
            self.renderDir = 1 # Anterior
        elif direction == 2: # Anterior to Left
            self.ren.GetActiveCamera().SetPosition(fp[0]+dist, fp[1], fp[2])
            self.ren.GetActiveCamera().SetViewUp(0.0, 0.0, 1.0)
            self.renderDir = 2 # Left
        elif direction == 3: # Left to Superior
            # from Head
            self.ren.GetActiveCamera().SetPosition(fp[0], fp[1], fp[2]+dist)
            self.ren.GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)
            self.renderDir = 3 # Superior

        self.ren.GetActiveCamera().ParallelProjectionOn()
        self.ren.GetActiveCamera().GetViewTransformMatrix()
        #print(self.ren.GetActiveCamera().GetViewTransformMatrix())
        self.ren.ResetCameraClippingRange()
        self.ren.ResetCamera()    
        self.widget.update()

    def renderMode(self,mode):
        print("Rendering Mode")
        opacityWindow = 2048 #(max_npV-min_npV)/4.0
        opacityLevel = 1024 #(max_npV-min_npV)/2.0

        mapper = self.volume.GetMapper()
        property = self.volume.GetProperty()
        colorFun = vtk.vtkColorTransferFunction()
        opacityFun = vtk.vtkPiecewiseFunction()
        gradientFun = vtk.vtkPiecewiseFunction()

        if mode == 2: # bone only
            colorFun.AddRGBPoint(self.min_npV, 0.3, 0.3, 1.0, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[0], 0.95, 0.95, 0.85, 0.5, 0.0)
            colorFun.AddRGBPoint((self.adjThresholds[0]+self.adjThresholds[2])/2, 0.75, 0.4, 0.35, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[2], .95, .84, .19, .5, 0.0)
            colorFun.AddRGBPoint(self.max_npV, 0.78, 0.78, 0.92, .5, 0.0)
      
            width_s=80
            opacityFun.AddPoint(self.min_npV, 0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            # Right
            opacityFun.AddPoint(self.adjThresholds[1], 0, .5, .0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            opacityFun.AddPoint(self.adjThresholds[2], 0.5, .5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            opacityFun.AddPoint(self.max_npV, .75, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            
            gradientFun.AddPoint(self.min_npV, 1.0, 0.5,.0)
            gradientFun.AddPoint(self.min_npV + (self.max_npV-self.min_npV)*0.2,.0,0.5,.0)
            #gradientFun.AddPoint(28.,.0,0.0,.0)
            gradientFun.AddPoint(self.max_npV,1.0,0.5,.0)

            property.ShadeOn()
            mapper.SetBlendModeToComposite()
            property.SetAmbient(0.2)
            property.SetDiffuse(1.0)
            property.SetSpecular(0.0)
            property.SetSpecularPower(1.0)
            property.SetScalarOpacityUnitDistance(0.8919)    
            self.rendermode = 2

        elif mode == 3: # MIP
            colorFun.AddRGBSegment(0.0, 1.0, 1.0, 1.0, 255.0, 1.0, 1.0, 1.0)
            opacityFun.AddSegment(opacityLevel - 0.5 * opacityWindow, 0.0, opacityLevel + 0.5 * opacityWindow, 1.0)
            mapper.SetBlendModeToMaximumIntensity()            
            self.rendermode = 3

        elif mode == 1: # Bone and soft tissue
            width_s=80
            width_l=160

            colorFun.AddRGBPoint(self.min_npV, 0.3, 0.3, 1.0, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[0], 0.95, 0.95, 0.85, 0.5, 0.0)
            colorFun.AddRGBPoint((self.adjThresholds[0]+self.adjThresholds[2])/2, 0.75, 0.4, 0.35, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[2], .95, .84, .19, .5, 0.0)
            colorFun.AddRGBPoint(self.max_npV, 0.78, 0.78, 0.92, .5, 0.0)
        
            opacityFun.AddPoint(self.min_npV, 0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            # Right
            opacityFun.AddPoint(self.adjThresholds[0], .0, .5, .0)
            opacityFun.AddPoint(self.adjThresholds[0]+width_s/2.0, 0.5, .5, .0)
            opacityFun.AddPoint(self.adjThresholds[0]+width_s, 0.0, .5, .0)
            opacityFun.AddPoint(self.adjThresholds[1], 0, .5, .0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            opacityFun.AddPoint(self.adjThresholds[2], 0.5, .5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            opacityFun.AddPoint(self.max_npV, .75, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            
            gradientFun.AddPoint(self.min_npV, 1.0, 0.5,.0)
            gradientFun.AddPoint(self.min_npV + (self.max_npV-self.min_npV)*0.2,.0,0.5,.0)
            #gradientFun.AddPoint(28.,.0,0.0,.0)
            gradientFun.AddPoint(self.max_npV,1.0,0.5,.0)

            property.ShadeOn()
            mapper.SetBlendModeToComposite()
            property.SetAmbient(0.2)
            property.SetDiffuse(1.0)
            property.SetSpecular(0.0)
            property.SetSpecularPower(1.0)
            property.SetScalarOpacityUnitDistance(0.8919)             
            self.rendermode = 1
        
        elif mode == 4: # bone only others
            colorFun.AddRGBPoint(self.min_npV, 0.3, 0.3, 1.0, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[0], 0.95, 0.95, 0.85, 0.5, 0.0)
            colorFun.AddRGBPoint((self.adjThresholds[1]+self.adjThresholds[2])/2, 0.75, 0.4, 0.35, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[3], .95, .84, .19, .5, 0.0)
            colorFun.AddRGBPoint(self.max_npV, 0.78, 0.78, 0.92, .5, 0.0)
      
            width_s=80
            opacityFun.AddPoint(self.min_npV, 0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            # Right
            opacityFun.AddPoint(self.adjThresholds[2], 0, .5, .0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            opacityFun.AddPoint(self.adjThresholds[3], 0.5, .5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            opacityFun.AddPoint(self.max_npV, .75, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            
            gradientFun.AddPoint(self.min_npV, 1.0, 0.5,.0)
            gradientFun.AddPoint(self.min_npV + (self.max_npV-self.min_npV)*0.2,.0,0.5,.0)
            #gradientFun.AddPoint(28.,.0,0.0,.0)
            gradientFun.AddPoint(self.max_npV,1.0,0.5,.0)

            property.ShadeOn()
            mapper.SetBlendModeToComposite()
            property.SetAmbient(0.2)
            property.SetDiffuse(1.0)
            property.SetSpecular(0.0)
            property.SetSpecularPower(1.0)
            property.SetScalarOpacityUnitDistance(0.8919)    
            self.rendermode = 4

        elif mode == 5: # soft tissue(right+)
            width_s=80
            width_l=160

            colorFun.AddRGBPoint(self.min_npV, 0.3, 0.3, 1.0, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[0], 0.95, 0.95, 0.85, 0.5, 0.0)
            colorFun.AddRGBPoint((self.adjThresholds[0]+self.adjThresholds[2])/2, 0.75, 0.4, 0.35, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[2], .95, .84, .19, .5, 0.0)
            colorFun.AddRGBPoint(self.max_npV, 0.78, 0.78, 0.92, .5, 0.0)
        
            opacityFun.AddPoint(self.min_npV, 0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            # Right
            opacityFun.AddPoint(self.adjThresholds[0], .0, .5, .0)
            opacityFun.AddPoint(self.adjThresholds[0]+width_s/2.0, 0.5, .5, .0)
            opacityFun.AddPoint(self.adjThresholds[0]+width_s, .0, .5, .0)
            opacityFun.AddPoint(self.max_npV, .0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            
            gradientFun.AddPoint(self.min_npV, 1.0, 0.5,.0)
            gradientFun.AddPoint(self.min_npV + (self.max_npV-self.min_npV)*0.2,.0,0.5,.0)
            #gradientFun.AddPoint(28.,.0,0.0,.0)
            gradientFun.AddPoint(self.max_npV,1.0,0.5,.0)

            property.ShadeOn()
            mapper.SetBlendModeToComposite()
            property.SetAmbient(0.2)
            property.SetDiffuse(1.0)
            property.SetSpecular(0.0)
            property.SetSpecularPower(1.0)
            property.SetScalarOpacityUnitDistance(0.8919)             
            self.rendermode = 5

        elif mode == 6: # soft tissue(middle)
            width_s=80
            width_l=160

            colorFun.AddRGBPoint(self.min_npV, 0.3, 0.3, 1.0, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[0], 0.95, 0.95, 0.85, 0.5, 0.0)
            colorFun.AddRGBPoint((self.adjThresholds[0]+self.adjThresholds[2])/2, 0.75, 0.4, 0.35, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[2], .95, .84, .19, .5, 0.0)
            colorFun.AddRGBPoint(self.max_npV, 0.78, 0.78, 0.92, .5, 0.0)
        
            opacityFun.AddPoint(self.min_npV, 0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            # Right
            opacityFun.AddPoint(self.adjThresholds[0]-width_s/2.0, .0, .5, .0)
            opacityFun.AddPoint(self.adjThresholds[0], 0.5, .5, .0)
            opacityFun.AddPoint(self.adjThresholds[0]+width_s/2.0, 0.0, .5, .0)
            opacityFun.AddPoint(self.max_npV, .0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            
            gradientFun.AddPoint(self.min_npV, 1.0, 0.5,.0)
            gradientFun.AddPoint(self.min_npV + (self.max_npV-self.min_npV)*0.2,.0,0.5,.0)
            #gradientFun.AddPoint(28.,.0,0.0,.0)
            gradientFun.AddPoint(self.max_npV,1.0,0.5,.0)

            property.ShadeOn()
            mapper.SetBlendModeToComposite()
            property.SetAmbient(0.2)
            property.SetDiffuse(1.0)
            property.SetSpecular(0.0)
            property.SetSpecularPower(1.0)
            property.SetScalarOpacityUnitDistance(0.8919)             
            self.rendermode = 6

        elif mode == 7: # soft tissue(left-)
            width_s=80
            width_l=160

            colorFun.AddRGBPoint(self.min_npV, 0.3, 0.3, 1.0, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[0], 0.95, 0.95, 0.85, 0.5, 0.0)
            colorFun.AddRGBPoint((self.adjThresholds[0]+self.adjThresholds[2])/2, 0.75, 0.4, 0.35, 0.5, 0.0)
            colorFun.AddRGBPoint(self.adjThresholds[2], .95, .84, .19, .5, 0.0)
            colorFun.AddRGBPoint(self.max_npV, 0.78, 0.78, 0.92, .5, 0.0)
        
            opacityFun.AddPoint(self.min_npV, 0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            # Right
            opacityFun.AddPoint(self.adjThresholds[0]-width_s, .0, .5, .0)
            opacityFun.AddPoint(self.adjThresholds[0]-width_s/2.0, 0.5, .5, .0)
            opacityFun.AddPoint(self.adjThresholds[0], 0.0, .5, .0)
            opacityFun.AddPoint(self.max_npV, .0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
            
            gradientFun.AddPoint(self.min_npV, 1.0, 0.5,.0)
            gradientFun.AddPoint(self.min_npV + (self.max_npV-self.min_npV)*0.2,.0,0.5,.0)
            #gradientFun.AddPoint(28.,.0,0.0,.0)
            gradientFun.AddPoint(self.max_npV,1.0,0.5,.0)

            property.ShadeOn()
            mapper.SetBlendModeToComposite()
            property.SetAmbient(0.2)
            property.SetDiffuse(1.0)
            property.SetSpecular(0.0)
            property.SetSpecularPower(1.0)
            property.SetScalarOpacityUnitDistance(0.8919)             
            self.rendermode = 7


        property.SetColor(colorFun)
        property.SetScalarOpacity(opacityFun)
        #property.SetGradientOpacity(gradientFun)
        self.widget.update()

    def captureScreen(self,studyUID):
        print("Capture")
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToWorld()
        vb = self.volume.GetBounds()
        extent = Extent(*vb)
        dispCoord = []
        for i, vertex in enumerate(extent.vertices()):
            coordinate.SetValue(vertex)
            dispCoord.append(coordinate.GetComputedDisplayValue(self.ren))
            print(i, vertex, dispCoord[i])

        winToImageFilter = vtk.vtkWindowToImageFilter()
        winToImageFilter.SetInput(self.widget.GetRenderWindow())
        winToImageFilter.SetInputBufferTypeToRGB()
        winToImageFilter.Update()
        vtk_data = winToImageFilter.GetOutput()

        imgArray = vtkmodules.util.numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetScalars())
        x,y = self.widget.GetRenderWindow().GetSize()
        imgArray = imgArray.reshape(x, y, 3)

        # clip image
        x,y = self.widget.GetRenderWindow().GetSize()
        if (self.renderDir==1):   # Anterior to Posterior
            x1 = dispCoord[0][0]
            y1 = dispCoord[0][1]
            x2 = dispCoord[5][0]
            y2 = dispCoord[5][1]
        elif (self.renderDir==2): # Left
            x1 = dispCoord[4][0]
            y1 = dispCoord[4][1]
            x2 = dispCoord[7][0]
            y2 = dispCoord[7][1]
        elif (self.renderDir==3): # Head
            x1 = dispCoord[1][0]
            y1 = dispCoord[1][1]
            x2 = dispCoord[7][0]
            y2 = dispCoord[7][1]

        if (x1<x-x2): dx= x1 
        else: dx = x-x2
        if (y1<y-y2): dy= y1 
        else: dy = y-y2

        croppedArray = imgArray[dy:y-dy+1, dx:x-dx+1, :]
        croppedArray = numpy.flipud(croppedArray)
        scrFileName = os.path.join(self.outdir,"%s_%d%d%d.png" % (studyUID, self.rendermode, self.renderDir, self.screenshotCount))
        txtFileName = os.path.join(self.outdir,"%s_%d%d%d.txt" % (studyUID, self.rendermode, self.renderDir, self.screenshotCount))

        print("Save to ",scrFileName)
        self.screenshotCount += 1

        xx= x-2*dx-1 #column
        yy= y-2*dy-1 #row

        if (self.wcs_extent):

            txtFile = open(txtFileName, "w")

            for j,seg_extent in enumerate(self.wcs_extent):
                sb = Extent(*seg_extent)
                dpCoord = []
                for i, vertex in enumerate(sb.vertices()):
                    coordinate.SetValue(vertex)
                    dpCoord.append(coordinate.GetComputedDisplayValue(self.ren))
                
                if (self.renderDir==1):   # Anterior to Posterior
                    xx1 = dpCoord[0][0]
                    yy1 = y-dpCoord[0][1]-1
                    xx2 = dpCoord[5][0]
                    yy2 = y-dpCoord[5][1]-1
                elif (self.renderDir==2): # Left
                    xx1 = dpCoord[4][0]
                    yy1 = y-dpCoord[4][1]-1
                    xx2 = dpCoord[7][0]
                    yy2 = y-dpCoord[7][1]-1

                elif (self.renderDir==3): # Head
                    xx1 = dpCoord[1][0]
                    yy1 = y-dpCoord[1][1]-1
                    xx2 = dpCoord[7][0]
                    yy2 = y-dpCoord[7][1]-1
                del dpCoord           

                print("Segment:",j)
                cx, cy = (xx1+xx2-2*dx)/(2*xx), (yy1+yy2-2*dy)/(2*yy)
                width, height = (xx2-xx1)/xx, (yy1-yy2)/yy
                print(x, y, dx, dy, xx, yy, xx1, yy1, xx2, yy2)
                print(xx1-dx, yy1-dy, xx2-dx, yy2-dy)
                print((xx1-dx)/xx, (yy1-dy)/yy, (xx2-dx)/xx, (yy2-dy)/yy)
                print(cx,cy,width,height)
                txtFile.write(f"{j} {cx} {cy} {width} {height}\n")

            txtFile.close()

            
        # wcs to disp, vol.bounds to crop, segmented bounds ratio(center(x,y), width, height), save to text file
        
        data = im.fromarray(croppedArray) 
        data.save(scrFileName)
            
    def traverse(self):
        files = os.listdir(self.rootpath)
        for i,file in enumerate(files):
            path = os.path.join(self.rootpath, file)
            if os.path.isdir(path):
                print(i,"-th start processing",path)
                self.capture(path)
                self.reset()        
                if (i==1): break

if __name__ == "__main__":
    app = QApplication(['View Relu Results'])
    
    root_directory = "c:\\dev\Testdata"

    files = os.listdir(root_directory)

    dcmRenderObject = fill_json(root_directory)
    print("Rendering object Created!")
    
    button = QPushButton("Traverse")
    button.clicked.connect(dcmRenderObject.traverse)
    button.show()

    sys.exit(app.exec())
            

