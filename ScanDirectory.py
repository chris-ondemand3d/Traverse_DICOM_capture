###########################################################################################
#
#  Program: Console app to scan directory, load dicom series, and render the series volume
#  using GDCM (Grassroots DICOM). A DICOM library ( > 2.8.6 version required by performance)
#
#  Copyright (c) 2006-2011 Mathieu Malaterre
#  All rights reserved.
#  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.
#
#     This software is distributed WITHOUT ANY WARRANTY; without even
#     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#     PURPOSE.  See the above copyright notice for more information.
#
###########################################################################################

import sys, os
import subprocess,time

import gdcm
import numpy
import vtk
import matplotlib.pyplot as plt
#import cv2
import SimpleITK as sitk
from scipy import ndimage
from scipy.signal import argrelextrema
from skimage import data
from skimage.filters import threshold_multiotsu

import dicom2nifti.convert_dir as convert_directory
import nibabel as nib
from scipy.ndimage import label
MinConnectedVoxel = 5000

class ProgressWatcher(gdcm.SimpleSubjectWatcher):
    def ShowProgress(self, sender, event):
        pe = gdcm.ProgressEvent.Cast(event)
        #print(pe.GetProgress())

    def EndFilter(self):
        pass  # print ("Yay ! I am done")

def get_gdcm_to_numpy_typemap():
    """Returns the GDCM Pixel Format to numpy array type mapping."""
    _gdcm_np = {gdcm.PixelFormat.UINT8: numpy.uint8,
                gdcm.PixelFormat.INT8: numpy.int8,
                # gdcm.PixelFormat.UINT12 :numpy.uint12,
                # gdcm.PixelFormat.INT12  :numpy.int12,
                gdcm.PixelFormat.UINT16: numpy.uint16,
                gdcm.PixelFormat.INT16: numpy.int16,
                gdcm.PixelFormat.UINT32: numpy.uint32,
                gdcm.PixelFormat.INT32: numpy.int32,
                # gdcm.PixelFormat.FLOAT16:numpy.float16,
                gdcm.PixelFormat.FLOAT32: numpy.float32,
                gdcm.PixelFormat.FLOAT64: numpy.float64}
    return _gdcm_np


def get_numpy_array_type(gdcm_pixel_format):
    """Returns a numpy array typecode given a GDCM Pixel Format."""
    return get_gdcm_to_numpy_typemap()[gdcm_pixel_format]


def gdcm_to_numpy(image):
    """Converts a GDCM image to a numpy array.
    """
    pf = image.GetPixelFormat()

    assert pf.GetScalarType() in get_gdcm_to_numpy_typemap().keys(), "Unsupported array type %s" % pf
    assert pf.GetSamplesPerPixel() == 1, "SamplesPerPixel is not 1" % pf.GetSamplesPerPixel()
    shape = image.GetDimension(0) * image.GetDimension(1)
    if image.GetNumberOfDimensions() == 3:
        shape = shape * image.GetDimension(2)

    dtype = get_numpy_array_type(pf.GetScalarType())
    gdcm_array = image.GetBuffer().encode("utf-8", errors="surrogateescape")
    volume = numpy.frombuffer(gdcm_array, dtype=dtype)

    if image.GetNumberOfDimensions() == 2:
        result = volume.reshape(image.GetDimension(0), image.GetDimension(1))
    elif image.GetNumberOfDimensions() == 3:
        result = volume.reshape(image.GetDimension(2), image.GetDimension(0), image.GetDimension(1))

    #    result.shape = shape
    return result

def toList(a):
	s = str(a)
	numbers = s.split('\\')
	float_numbers = [float(num) for num in numbers]
	return float_numbers

def getKey(item):
    return item[2]


# for multiframe
def GetSpacingDirOrigin(ds):
    sSFG = gdcm.Tag(0x5200, 0x9229) # Shared Functional Group 
    sPFFG = gdcm.Tag(0x5200,0x9230) # Per frame Functional Group
    if ds.FindDataElement( sSFG ):		
        sis = ds.GetDataElement( sSFG )
        sqsis = sis.GetValueAsSQ()
        if sqsis.GetNumberOfItems():
            item1 = sqsis.GetItem(1)
            nestedds = item1.GetNestedDataSet()
            sPOS = gdcm.Tag(0x0020,0x9116) # Plane Orientation Sequence
            sPMS = gdcm.Tag(0x0028,0x9110) # Pixel Measure Sequence
            if nestedds.FindDataElement( sPOS ):
                prcs = nestedds.GetDataElement( sPOS )
                sqprcs = prcs.GetValueAsSQ()
                if sqprcs.GetNumberOfItems():
                    item2 = sqprcs.GetItem(1)
                    nestedds2 = item2.GetNestedDataSet()
                    sIOP = gdcm.Tag(0x0020,0x0037) #Image Orientation Patient
                    if nestedds2.FindDataElement( sIOP ):
                        cm = nestedds2.GetDataElement( sIOP )
                        imageOrientation = toList(cm.GetValue())
                        bIO = True
                        print("Image Orientation",imageOrientation)
                    else:
                        bIO = False
                        print("No Image Orientation")        
            if nestedds.FindDataElement( sPMS ):
                prcs = nestedds.GetDataElement( sPMS )
                sqprcs = prcs.GetValueAsSQ()
                if sqprcs.GetNumberOfItems():
                    item2 = sqprcs.GetItem(1)
                    nestedds2 = item2.GetNestedDataSet()
                    sPS = gdcm.Tag(0x0028,0x0030) #Pixel Spacing
                    if nestedds2.FindDataElement( sPS ):
                        cm = nestedds2.GetDataElement( sPS )
                        pixelSpacing = toList(cm.GetValue())
                        bPS = True
                        print("PixelSpacing",pixelSpacing)
                    else:
                        bPS = False
                        print("No Pixel Spacing")
                    sST = gdcm.Tag(0x0018,0x0050) # SLice Thickness
                    if nestedds2.FindDataElement( sST ):
                        cm = nestedds2.GetDataElement( sST )
                        sliceThickness = toList(cm.GetValue())
                        bST = True
                        print("SliceThickness",sliceThickness)
                    else:
                        bST = False
                        print("No Slice Thickness")

    if ds.FindDataElement( sPFFG ):
        sis = ds.GetDataElement( sPFFG )
        sqsis = sis.GetValueAsSQ()
        nFrame = sqsis.GetNumberOfItems()

        item = sqsis.GetItem(1)
        nestedds = item.GetNestedDataSet()
        sPPS = gdcm.Tag(0x0020,0x9113) # Plane Position Sequence
        if nestedds.FindDataElement( sPPS ):
            prcs = nestedds.GetDataElement( sPPS )
            sqprcs = prcs.GetValueAsSQ()
            if sqprcs.GetNumberOfItems():
                item2 = sqprcs.GetItem(1)
                nestedds2 = item2.GetNestedDataSet()
                sIP = gdcm.Tag(0x0020,0x0032)
                if nestedds2.FindDataElement( sIP ):
                    cm = nestedds2.GetDataElement( sIP )
                    pos0 = toList(cm.GetValue())
                    print("ImagePosition",pos0)
                else:
                    print(i,"No Image Position")

        item = sqsis.GetItem(nFrame)
        nestedds = item.GetNestedDataSet()
        sPPS = gdcm.Tag(0x0020,0x9113) # Plane Position Sequence
        if nestedds.FindDataElement( sPPS ):
            prcs = nestedds.GetDataElement( sPPS )
            sqprcs = prcs.GetValueAsSQ()
            if sqprcs.GetNumberOfItems():
                item2 = sqprcs.GetItem(1)
                nestedds2 = item2.GetNestedDataSet()
                sIP = gdcm.Tag(0x0020,0x0032)
                if nestedds2.FindDataElement( sIP ):
                    cm = nestedds2.GetDataElement( sIP )
                    bOri = True
                    pos1 = toList(cm.GetValue())
                    print("ImagePosition",pos1)
                else:
                    print("No Image Position")
                    bOri = False
    if (bIO and bOri):
        cosineX = numpy.array(imageOrientation[0:3])
        cosineY = numpy.array(imageOrientation[3:6])
        normal = numpy.cross(cosineX, cosineY)
        posV0 = numpy.array(pos0)
        posV1 = numpy.array(pos1)
        dist = (numpy.dot(normal,posV1)-numpy.dot(normal,posV0))/(nFrame-1)
        if (dist<0):
            bFlip = True
            origin = pos1
        else:
            bFlip = False
            origin= pos0
        return imageOrientation,sliceThickness,origin,pixelSpacing,abs(dist),bFlip,pos0,pos1        
    else:
        return -1

def GetZSpacing(dataset, tag, directionCosines):
    if (not dataset.FindDataElement(tag)):
        return -1
    sqi = dataset.GetDataElement(tag).GetValueAsSQ()
    nItems = sqi.GetNumberOfItems()
    if (not sqi or nItems == 0):
        return -2
    cosineX = numpy.array(directionCosines[0:3])
    cosineY = numpy.array(directionCosines[3:6])
    normal = numpy.cross(cosineX, cosineY)
    dist = numpy.zeros(nItems)

    for i in range(nItems):
        # print(i+1, "th item: in ", nItems, ":")
        item = sqi.GetItem(i + 1)
        subds = item.GetNestedDataSet()

        # Plane Position Sequence
        tpms = gdcm.Tag(0x0020, 0x9113)
        if (not subds.FindDataElement(tpms)):
            return -3

        sqi2 = subds.GetDataElement(tpms).GetValueAsSQ()
        if (not sqi2 or sqi2.GetNumberOfItems() == 0):
            return -4

        item2 = sqi2.GetItem(1)
        subds2 = item2.GetNestedDataSet()

        tps = gdcm.Tag(0x0020, 0x0032)
        if (not subds2.FindDataElement(tps)):
            print("Not exist 0020,0032")

        de2 = subds2.GetDataElement(tps)
        posV = toList(str(de2.GetValue()))
        pos = numpy.array(posV)
        dist[i] = numpy.dot(normal, pos)

    prev = dist[0]
    sum = 0
    for i in range(nItems - 1):
        sum = sum + (dist[i + 1] - prev)
        prev = dist[i + 1]

    return sum / (nItems - 1)


def numpy2VTK(img, spacing=[1.0, 1.0, 1.0], origin=[0.0,0.0,0.0],dirCosines=[1.0,0.0,0.0,0.0,1.0,0.0]):
    # evolved from code from Stou S.,
    # on http://www.siafoo.net/snippet/314
    # This function, as the name suggests, converts numpy array to VTK
    # Check origin, direction, spacing

    importer = vtk.vtkImageImport()

    img_data = img.astype('int16')
    img_string = img_data.tobytes()  # type short
    dim = img.shape

    # vtkData = numpy_support.numpy_to_vtk(num_array=img_data.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_INT)

    print(len(img_string), dim)
    # for i in range(100):
    #    prStr = ""
    #    for j in range(100):
    #       prStr += ' ' + str(img_string[(100+i)*375*375 + (100+j)*375 + 100])
    #    print(prStr)

    importer.CopyImportVoidPointer(img_string, len(img_string))  # (dim[0]*dim[1]*dim[2])
    importer.SetDataScalarType(vtk.VTK_SHORT)
    importer.SetNumberOfScalarComponents(1)

    extent = importer.GetDataExtent()
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[0] - 1)

    importer.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin(origin[0],origin[1],origin[2])
    cosX = numpy.array(dirCosines[0:3])
    cosY = numpy.array(dirCosines[3:6])
    cosZ = numpy.cross(cosX, cosY)
    R=numpy.transpose([cosX,cosY,cosZ])
    importer.SetDataDirection(R.flatten()) 

    return importer


def show_mid_slice(img_numpy, title='img'):
   """
   Accepts an 3D numpy array and shows median slices in all three planes
   """
   assert img_numpy.ndim == 3
   n_i, n_j, n_k = img_numpy.shape

   # sagittal (left image)
   center_i1 = int((n_i - 1) / 2)
   # coronal (center image)
   center_j1 = int((n_j - 1) / 2)
   # axial slice (right image)
   center_k1 = int((n_k - 1) / 2)

   show_slices([img_numpy[center_i1, :, :],
                img_numpy[:, center_j1, :],
                img_numpy[:, :, center_k1]])
   plt.suptitle(title)

def show_slices(slices):
   """
   Function to display a row of image slices
   Input is a list of numpy 2D image slices
   """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice, cmap="gray", origin="lower")
   plt.show()

def find_connected_elements_3d(m, v):
    # Create a binary mask where elements equal to v are True, others are False
    mask = (m == v)    
    # Use scipy's label function to identify connected components
    labeled_array, num_features = label(mask)
    # Get the indices for each labeled region
    indices = [numpy.argwhere(labeled_array == i) for i in range(1, num_features + 1)]
    return indices

def load_dicom(directory,blendType=3,makenifti=True):

    # Define the set of tags we are interested in, may need more
    t1 = gdcm.Tag(0x10, 0x20);  # Patient ID
    t2 = gdcm.Tag(0x10, 0x10);  # Patient Name
    t3 = gdcm.Tag(0x20, 0x10);  # Study ID
    t4 = gdcm.Tag(0x20, 0x0d);  # Study Instance UID
    t5 = gdcm.Tag(0x20, 0x0e);  # Series Instance UID
    t6 = gdcm.Tag(0x20, 0x11);  # Series Number
    t7 = gdcm.Tag(0x28, 0x08);  # Number of Frames
    t8 = gdcm.Tag(0x20, 0x32);  # Image Position
    t10 = gdcm.Tag(0x28, 0x30);  # Pixel Spacing
    t11 = gdcm.Tag(0x20, 0x37);  # Image Orientation Patient
    t12 = gdcm.Tag(0x28, 0x02);  # Samples per pixel
    t13 = gdcm.Tag(0x28, 0x04);  # Photometric Interpretation
    t14 = gdcm.Tag(0x28, 0x10);  # Rows
    t15 = gdcm.Tag(0x28, 0x11);  # Column
    t16 = gdcm.Tag(0x28, 0x101);  # BitStored
    t17 = gdcm.Tag(0x02, 0x02);  # Media Storage SOP Class UID
    t18 = gdcm.Tag(0x02, 0x03);  # Media Storage SOP Instance UID
    t19 = gdcm.Tag(0x02, 0x10);  # Transfer Syntax
    t20 = gdcm.Tag(0x08, 0x16);  # SOP Class UID
    t21 = gdcm.Tag(0x08, 0x18);  # SOP Instance UID
    t22 = gdcm.Tag(0x5200, 0x9229);  # Shared functional group
    t23 = gdcm.Tag(0x5200, 0x9230);  # Per frame functional group
    t24 = gdcm.Tag(0x0028, 0x1050);  # WindowCenter
    t25 = gdcm.Tag(0x0028, 0x1051);  # WindowWidth
    t26 = gdcm.Tag(0x0028, 0x1052);  # Rescale Intercept
    t27 = gdcm.Tag(0x0028, 0x1053);  # Rescale Slope
    t28 = gdcm.Tag(0x0028, 0x1054);  # Rescale Type

    # for profiling
    currentTime = time.time()

    # Iterate over directory
    d = gdcm.Directory();
    nfiles = d.Load(directory);
    if (nfiles == 0): return None

    filenames = d.GetFilenames()
    # print("Files ", filenames)

    #  Get rid of any Warning while parsing the DICOM files
    gdcm.Trace.WarningOff()

    # instanciate Scanner:
    sp = gdcm.Scanner.New();
    s = sp.__ref__()
    # w = ProgressWatcher(s, 'Watcher')

    s.AddTag(t1);
    s.AddTag(t2);
    s.AddTag(t3);
    s.AddTag(t4);
    s.AddTag(t5);
    s.AddTag(t6);
    s.AddTag(t7);
    s.AddTag(t8);
    s.AddTag(t10);
    s.AddTag(t11);
    s.AddTag(t12);
    s.AddTag(t13);
    s.AddTag(t14);
    s.AddTag(t15);
    s.AddTag(t16);
    s.AddTag(t17);
    s.AddTag(t18);
    s.AddTag(t19);
    s.AddTag(t20);
    s.AddTag(t21);
    s.AddTag(t22);
    s.AddTag(t23);

    b = s.Scan(filenames);
    if (not b): sys.exit(1);
    print("success", b);

    # for profiling
    print("Time to Scan Directory and Dicom files", time.time() - currentTime)
    currentTime = time.time()

    dicomfiles = []
    patient_list = []
    study_list = []
    series_list = []

    for dFile in filenames:
        if (s.IsKey(dFile)):  # existing DICOM file

            #print(dFile)
            is_multiframe = 0

            pttv = gdcm.PythonTagToValue(s.GetMapping(dFile))
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
                    if (value not in patient_list): patient_list.append(value)
                    patient_id = value
                elif (tag == t2):
                    # print ("PatientName->",value)
                    pass
                elif (tag == t3):
                    # print ("StudyID->",value)
                    pass
                elif (tag == t4):
                    # print ("StudyInstanceUID->",value)
                    if (value not in study_list): study_list.append(value)
                    study_id = value
                elif (tag == t6):
                    # print ("SeriesNum->",value)
                    pass
                elif (tag == t5):
                    # print ("SeriesInstanceUID->",value)
                    if (value not in series_list): series_list.append(value)
                    series_id = value
                elif (tag == t7):
                    # print ("NumberOfFrame->",value)
                    if (int(value) > 1):
                        is_multiframe = int(value)
                    else:
                        is_multiframe = 0
                elif (tag == t8):
                    #print("Image Patient Position->",value)
                    pass
                elif (tag == t19):
                    # print("Transfer Syntax->",value)
                    pass
                elif (tag == t20):
                    # print("SOP Class UID->",value)
                    pass
                elif (tag == t21):
                    # print("SOP Instance UID->",value)
                    instance_id = value
                    pass
                elif (tag == t22):
                    #print("Shared Functional Group Sequence->",value)
                    pass
                # increment iterator
                pttv.Next()

                # dicomfiles.append('PatientID':patient_id,'StudyID':study_id, 'SeriesID':seriesID, 'Multiframe':is_multiframe)
            if (not study_id): print("Missing StudyUID ")
            if (not series_id): print("Missing SeriesUID")

            dicomfiles.append(
                {'PatientID': patient_id, 'StudyID': study_id, 'SeriesID': series_id, 'Multiframe': is_multiframe,
                 'InstanceUID': instance_id, 'FileName': dFile})
            
    if (len(series_list)>1 and len(series_list)==0): 
        return None # many series or no series
    else:     
        series_uid = series_list[0]

        series_imgfiles = s.GetAllFilenamesFromTagToValue(t5, series_uid)
        print("-----------------------------------------------------------------------")
        print(series_uid, ", # of Files: ", len(series_imgfiles))

        if (len(series_imgfiles) == 1):  # Single file series-> multiframe, or just single image
            # check multiframe
            image_file = series_imgfiles[0]
            nFrame = int(s.GetValue(image_file, t7))

            # check scout image

            if (nFrame > 1): # Multiframe Image
                print("Read Multiframe", image_file, nFrame)

                reader = gdcm.ImageReader()
                reader.SetFileName(image_file)
                if (not reader.Read()):
                    print("Cannot read image", image_file)
                    return None
                else:
                    image = reader.GetImage()
                    npVolume = gdcm_to_numpy(image).copy()
                    print("Shape of Volume:", npVolume.shape)

                    # Image Dimension
                    w, d, h = image.GetDimension(0), image.GetDimension(1), image.GetDimension(2)

                    # Get DirectionCosines, Origin, Spacing
                    f = reader.GetFile()
                    ds = f.GetDataSet()
                    
                    dirCosines, sliceThickness, origin, pixelSpacing, dz, bFlip, pos0, pos1 = GetSpacingDirOrigin(ds)
                    print(dirCosines, sliceThickness, origin, pixelSpacing, dz, bFlip) 
                    dx = pixelSpacing[0]
                    dy = pixelSpacing[1]
                    if (bFlip):
                        npVolume = numpy.flip(npVolume,0)

                    max_npV = npVolume.max()
                    min_npV = npVolume.min()

                    print("Samples per Pixel:", s.GetValue(image_file, t12))
                    print("Photometric Representation:", s.GetValue(image_file, t13))
                    print("Rows:", s.GetValue(image_file, t14))
                    print("Columns:", s.GetValue(image_file, t15))
                    print("BitStored:", s.GetValue(image_file, t16))

                    # f = reader.GetFile()
                    # ds = f.GetDataSet()
                    # print(GetZSpacing(ds, t23, cosines))

                    # CT default value
                    iRescaleSlope = 1.0
                    iRescaleIntercept = -1024.0
                    iWindowCenter = 1024
                    iWindowWidth = 4092

                    print("RescaleSlope:", s.GetValue(image_file, t27))
                    print("RescaleIntercept:", s.GetValue(image_file, t26))
                    #print("RescaleType:", s.GetValue(image_file, t28))
                    print("WindowCenter:", s.GetValue(image_file, t24))
                    print("WindowWidth:", s.GetValue(image_file, t25))

                    s27 = s.GetValue(image_file, t27)
                    s26 = s.GetValue(image_file, t26)
                    # s28 = s.GetValue(image_file, t28)
                    s24 = s.GetValue(image_file, t24)
                    s25 = s.GetValue(image_file, t25)

                    if s24: iWindowCenter = s24
                    if s25: iWindowWidth = s25
                    if s26: iRescaleIntercept = s26
                    if s27: iRescaleSlope = s27

            else:  # nFrame == 1
                print("Single image file with unique series id", image_file, nFrame)
                reader = gdcm.ImageReader()
                reader.SetFileName(image_file)
                if (not reader.Read()):
                    print("Cannot read image", image_file)
                    return None
                else:
                    npVolume = gdcm_to_numpy(reader.GetImage())
                    print("Shape of Volume:", npVolume.shape)
                    return None

        else:  # multiple files in a series
            # Read Postion and sorting
            series_files = []
            for i in range(len(series_imgfiles)):
                # convert ImagePosition
                strIP = s.GetValue(series_imgfiles[i], t8)
                posV = toList(strIP)
                series_files.append([series_imgfiles[i], posV])

            origin = series_files[0][1]
            pixelSpacing = toList(s.GetValue(series_files[0][0], t10))
            dirCosines = toList(s.GetValue(series_files[0][0], t11))
            rowCosine = numpy.array(dirCosines[:3])
            colCosine = numpy.array(dirCosines[3:])
            sliceCosine = numpy.cross(rowCosine,colCosine)

            for i in range(len(series_files)):
                series_files[i].append([numpy.dot((numpy.array(series_files[i][1])-numpy.array(origin)),sliceCosine)])

            sorted_series_files = sorted(series_files, key=getKey, reverse=False)

            if len(series_files)>1:
                slice_distances = []              
                for i in range(1,len(sorted_series_files)):
                    distance = numpy.linalg.norm(numpy.array(sorted_series_files[i][1])-numpy.array(sorted_series_files[i-1][1]))
                    slice_distances.append(distance)      
                avg_slice_distance = numpy.mean(slice_distances)
            else:
                avg_slice_distance = None

            dx = pixelSpacing[0]
            dy = pixelSpacing[1]            
            dz = avg_slice_distance

            reader = gdcm.ImageReader()
            reader.SetFileName(sorted_series_files[0][0])
            if (not reader.Read()):
                print("Cannot read image", sorted_series_files[0][0])

            image = reader.GetImage()
            pf = image.GetPixelFormat()
            assert pf.GetScalarType() in get_gdcm_to_numpy_typemap().keys(), "Unsupported array type %s" % pf
            #assert pf.GetSamplesPerPixel() == 1, "Support only one samples"
            w, d, h = image.GetDimension(0), image.GetDimension(1), len(sorted_series_files)
            spacing = image.GetSpacing()

            dtype = get_numpy_array_type(pf.GetScalarType())
            npVolume = numpy.zeros((h, w, d), dtype=dtype)
            print(w, d, h, dtype, dx, dy, dz)

            for i in range(h):
                reader = gdcm.ImageReader()
                reader.SetFileName(sorted_series_files[i][0])
                if (not reader.Read()):
                    print("Cannot read image", sorted_series_files[i][0])

                image = reader.GetImage()
                gdcm_array = image.GetBuffer().encode("utf-8", errors="surrogateescape")
                result = numpy.frombuffer(gdcm_array, dtype=dtype)
                npVolume[i, :, :] = result.reshape(w, d).copy() #numpy.flipud(result.reshape(d, w).copy())

            # Load images to numpy
            # 1st file에서 image plane, pixel정보 추출
            print("Pixel Spacing:", pixelSpacing)
            print("Image Orientation:", dirCosines)
            print("Origin:",origin)
            print("Samples per Pixel:", s.GetValue(sorted_series_files[0][0], t12))
            print("Photometric Representation:", s.GetValue(sorted_series_files[0][0], t13))
            print("Rows:", s.GetValue(sorted_series_files[0][0], t14))
            print("Columns:", s.GetValue(sorted_series_files[0][0], t15))
            print("BitStored:", s.GetValue(sorted_series_files[0][0], t16))

            max_npV = npVolume.max()
            min_npV = npVolume.min()

            # default value
            iRescaleSlope = 1
            if (min_npV < 0):
                iRescaleIntercept = 0
            else:
                iRescaleIntercept = -1024

            iWindowCenter = 1024
            iWindowWidth = 4092

            print("RescaleSlope:", s.GetValue(sorted_series_files[0][0], t27))
            print("RescaleIntercept:", s.GetValue(sorted_series_files[0][0], t26))
            print("RescaleType:", s.GetValue(sorted_series_files[0][0], t28))
            print("WindowCenter:", s.GetValue(sorted_series_files[0][0], t24))
            print("WindowWidth:", s.GetValue(sorted_series_files[0][0], t25))

            s27 = s.GetValue(sorted_series_files[0][0], t27)
            s26 = s.GetValue(sorted_series_files[0][0], t26)
            # s28 = s.GetValue(sorted_series_files[0][0], t28)
            s24 = s.GetValue(sorted_series_files[0][0], t24)
            s25 = s.GetValue(sorted_series_files[0][0], t25)

            if s24: iWindowCenter = s24
            if s25: iWindowWidth = s25
            if s26: iRescaleIntercept = s26
            if s27: iRescaleSlope = s27

            if s26 or (max_npV-min_npV)>= 4096:
                print("16bit data and rescale intercept", s26, s27)

        direction_x = dirCosines[0:3]
        direction_y = dirCosines[3:6]
        direction_z = numpy.cross(direction_x,direction_y)

        # pyplot single slice
        x = numpy.arange(0.0, (w+1)*dx, dx)
        y = numpy.arange(0.0, (d+1)*dy, dy)
        z = numpy.arange(0.0, (h+1)*dz, dz)
        # show_mid_slice(npVolume,title="CenterSlice")
        # show_slices([npVolume[10,:,:],npVolume[70,:,:],npVolume[120,:,:]])

        print("min, max, rescaleSlope, Intercept: ",npVolume.min(),npVolume.max(), iRescaleSlope, iRescaleIntercept)
    
        # direction = numpy.direction_x,direction_y,direction_z
        # print([dx, dy, dz], origin, direction_x,direction_y, direction_z)

    # vtk data importer
    # For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    dataImporter = numpy2VTK(npVolume, [dx, dy, dz], origin, dirCosines)
    dataImporter.Update()

    # Test for Removing some outlier, [npV_min, npV_min + 0.05*(max_npV-min_npV)]->0-rescale intercept, [npV_max-0.05*(max_npV-min_npV) ,npV_max] 값을 조정 
    npVolume[npVolume < (min_npV + 0.01*(max_npV-min_npV))] = 0

    start_time = time.time()

    npVol = npVolume*(256.0/(max_npV-min_npV))

    thresholds = threshold_multiotsu(npVol,4)
    end_time = time.time()
    adjThresholds = []
    for i in thresholds:
        #print(i*((max_npV-min_npV)/256.0), i*16, (i*16)*iRescaleSlope+iRescaleIntercept) 
        # original CT value, 4096 scale, applying rescale
        adjThresholds.append(i*((max_npV-min_npV)/256.0)*iRescaleSlope+iRescaleIntercept)
    print(adjThresholds, end_time-start_time)

    # makenifti files with directory, output_directory
    if makenifti:

        start_time = time.time()

        # parent_dir = os.path.dirname(os.path.abspath(directory))
        output_directory = os.path.join(directory,"segment")
        base_filename = "Dental_0001_0000"
        nifti_file = os.path.join(output_directory, base_filename + '.nii.gz')
        os.makedirs(output_directory, exist_ok=True)
        npImage = npVolume.transpose(2,1,0)

        step = [0,0,0]
        # TODO in case of multivolume
        if is_multiframe != 0:
            for i in range(3):
                step[i] = (pos0[i] - pos1[i])/(1-h)

        else:
            pos0 = sorted_series_files[0][1]
            pos1 = sorted_series_files[h-1][1]
            for i in range(3):
                step[i] = (pos0[i] - pos1[i])/(1-h)

        affine = numpy.array(
        [[-direction_x[0] * dy, -direction_y[0] * dx, -step[0], -origin[0]],
         [-direction_x[1] * dy, -direction_y[1] * dx, -step[1], -origin[1]],
         [direction_x[2] * dy, direction_y[2] * dx, step[2], origin[2]],
         [0, 0, 0, 1]])


        nii_image = nib.Nifti1Image(npImage, affine)
        nii_image.header.set_slope_inter(1, 0)
        nii_image.header.set_xyzt_units(2)  # set units for xyz (leave t as unknown)
        # nii_image.to_filename(nifti_file)
        nib.save(nii_image, nifti_file)

        # Segmentation  
        # Create input_folder and output_folder -> current folder - segment, output
        input_dir = os.path.join(directory,'segment')
        output_dir = os.path.join(directory,'output')
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(output_dir) and os.path.exists(input_dir):
            start_time = time.time()

            os.environ['nnUNet_raw'] = '.'
            os.environ['nnUNet_results'] = '.'
            os.environ['nnUNet_preprocessed'] = '.'

            # Command to run the nnUNetv2_predict script
            nnunet_command = (
                f"nnUNetv2_predict -i {input_dir}/ -o {output_dir}/ "
                "-d Dataset111_453CT -tr nnUNetTrainer -p nnUNetPlans "
                "-c 3d_fullres -f 0 -npp 1 -nps 1 -step_size 0.5 -device cuda --disable_tta"
            )
            # Execute the command
            subprocess.run(nnunet_command, shell=True) #, executable="/bin/bash"
        
            # 1,2,3,4,5각각에 대해서, ijk extent -> xyz extent in world coordinate 
            outfile = os.path.join(output_dir,"Dental_0001.nii.gz")
            if os.path.exists(outfile):
                end_time = time.time()
                print("segmentation time:", end_time-start_time)

                img = nib.load(outfile)            
                npImage = (img.get_fdata()).transpose(2,1,0)
                unique_labels = numpy.unique(npImage)
                
                avg = adjThresholds[2]
                for i in range(1,3):
                    if (i in unique_labels): # not segmented
                        indices = numpy.where(npImage == i)
                        values = npVolume[indices]
                        min_val = numpy.min(values)
                        max_val = numpy.max(values)
                        avg_val = numpy.mean(values)
                        var_val = numpy.var(values)
                        count = numpy.count_nonzero(values)
                        if (avg_val>avg): avg = avg_val
                        print(i,min_val,max_val,avg_val,var_val,count)
                adjThresholds.append(avg*iRescaleSlope+iRescaleIntercept)

    (xMin, xMax, yMin, yMax, zMin, zMax) = dataImporter.GetWholeExtent()
    (xSpacing, ySpacing, zSpacing) = dataImporter.GetDataSpacing()
    (x0, y0, z0) = dataImporter.GetDataOrigin()
    center = [x0 + xSpacing * 0.5 * (xMin + xMax),
              y0 + ySpacing * 0.5 * (yMin + yMax),
              z0 + zSpacing * 0.5 * (zMin + zMax)]
    
    print(xMin, xMax, yMin, yMax, zMin, zMax, xSpacing, ySpacing, zSpacing, x0,y0,z0, center)
    #print(iRescaleSlope, iRescaleIntercept, iWindowCenter, iWindowWidth)

    shiftScale = vtk.vtkImageShiftScale()
    shiftScale.SetScale(iRescaleSlope)
    shiftScale.SetShift(iRescaleIntercept)
    shiftScale.SetOutputScalarTypeToInt()
    shiftScale.ClampOverflowOn()
    shiftScale.SetInputConnection(dataImporter.GetOutputPort())
    shiftScale.Update()

    volume = vtk.vtkVolume()
    #mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    mapper.SetInputConnection(shiftScale.GetOutputPort())
    
   
    rotation_matrix = numpy.column_stack((rowCosine, colCosine, sliceCosine))
    # Create the 4x4 transformation matrix
    transform_matrix = numpy.eye(3)
    transform_matrix[:3, :3] = rotation_matrix.T
    
    mat = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            mat.SetElement(i,j,transform_matrix[i,j])
        mat.SetElement(i,3,-origin[i])

    volume.SetUserMatrix(mat)
   
    # max_npV, min_npV, adjThresholds

    # Transfer Function
    colorFun = vtk.vtkColorTransferFunction()
    opacityFun = vtk.vtkPiecewiseFunction()
    gradientFun = vtk.vtkPiecewiseFunction()
    # Create the property and attach the transfer functions
    property = vtk.vtkVolumeProperty()
    # property.SetIndependentComponents(independentComponents);
    property.SetColor(colorFun)
    property.SetScalarOpacity(opacityFun)
    #property.SetGradientOpacity(gradientFun)
    property.SetInterpolationTypeToLinear()
    # Try other volume rendering options
    
    opacityWindow = 2048 #(max_npV-min_npV)/4.0
    opacityLevel = 1024 #(max_npV-min_npV)/2.0

    if (blendType == 0):  # MIP
        colorFun.AddRGBSegment(0.0, 1.0, 1.0, 1.0, 255.0, 1.0, 1.0, 1.0)
        opacityFun.AddSegment(opacityLevel - 0.5 * opacityWindow, 0.0, opacityLevel + 0.5 * opacityWindow, 1.0)
        mapper.SetBlendModeToMaximumIntensity()
    elif (blendType == 1):  # ShadeOff
        colorFun.AddRGBSegment(opacityLevel - 0.5 * opacityWindow, 0.0, 0.0, 0.0, opacityLevel + 0.5 * opacityWindow,
                               1.0, 1.0, 1.0)
        opacityFun.AddSegment(opacityLevel - 0.5 * opacityWindow, 0.0, opacityLevel + 0.5 * opacityWindow, 1.0)
        mapper.SetBlendModeToComposite()
        property.ShadeOff()
    elif (blendType == 2):  # Shade On
        colorFun.AddRGBSegment(opacityLevel - 0.5 * opacityWindow, 0.0, 0.0, 0.0, opacityLevel + 0.5 * opacityWindow,1.0, 1.0, 1.0)
        opacityFun.AddSegment(opacityLevel - 0.5 * opacityWindow, 0.0, opacityLevel + 0.5 * opacityWindow, 1.0)
        mapper.SetBlendModeToComposite()
        property.ShadeOn()
    elif (blendType == 3):  # CT Bone1
        """# for 1st threshold, (width: 80), left, center,right
        # point1, adjThresholds[0], O = 0
        # point2, adjThresholds[0]+width/2, O=0.5, 0.25
        # point3, adjThresholds[0]+width, O = 0
        # for 1st threshold, (width: 160), left, center,right
        # point4, adjThresholds[2]-width/2, O=0
        # point5, adjThresholds[2]+width/2, O=0.5
        # point6, 3072, O=0.5 """

        colorFun.AddRGBPoint(min_npV*iRescaleSlope+iRescaleIntercept, 0.3, 0.3, 1.0, 0.5, 0.0)
        colorFun.AddRGBPoint(adjThresholds[0], 0.95, 0.95, 0.85, 0.5, 0.0)
        colorFun.AddRGBPoint((adjThresholds[0]+adjThresholds[2])/2, 0.75, 0.4, 0.35, 0.5, 0.0)
        colorFun.AddRGBPoint(adjThresholds[2], .95, .84, .19, .5, 0.0)
        colorFun.AddRGBPoint(max_npV*iRescaleSlope+iRescaleIntercept, 0.78, 0.78, 0.92, .5, 0.0)

        
        width_s=80
        width_l=160
        opacityFun.AddPoint(min_npV*iRescaleSlope+iRescaleIntercept, 0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
        # Right
        opacityFun.AddPoint(adjThresholds[0], .0, .5, .0)
        opacityFun.AddPoint(adjThresholds[0]+width_s/2.0, 0.5, .5, .0)
        opacityFun.AddPoint(adjThresholds[0]+width_s, 0.0, .5, .0)
        opacityFun.AddPoint(adjThresholds[1], 0, .5, .0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
        opacityFun.AddPoint(adjThresholds[2], 0.5, .5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
        opacityFun.AddPoint(max_npV*iRescaleSlope+iRescaleIntercept, .75, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
        
        
        gradientFun.AddPoint(min_npV*iRescaleSlope+iRescaleIntercept, 1.0, 0.5,.0)
        gradientFun.AddPoint(min_npV*iRescaleSlope+iRescaleIntercept + (max_npV-min_npV)*0.2,.0,0.5,.0)
        #gradientFun.AddPoint(28.,.0,0.0,.0)
        gradientFun.AddPoint(max_npV*iRescaleSlope+iRescaleIntercept,1.0,0.5,.0)

        property.ShadeOn()
        mapper.SetBlendModeToComposite()
        property.SetAmbient(0.2)
        property.SetDiffuse(1.0)
        property.SetSpecular(0.0)
        property.SetSpecularPower(1.0)
        property.SetScalarOpacityUnitDistance(0.8919)

    elif (blendType == 5): # CT Bone2
        colorFun.AddRGBPoint(min_npV*iRescaleSlope+iRescaleIntercept, 0.3, 0.3, 1.0, 0.5, 0.0)
        colorFun.AddRGBPoint(adjThresholds[0], 0.95, 0.95, 0.85, 0.5, 0.0)
        colorFun.AddRGBPoint((adjThresholds[0]+adjThresholds[2])/2, 0.75, 0.4, 0.35, 0.5, 0.0)
        colorFun.AddRGBPoint(adjThresholds[2], .95, .84, .19, .5, 0.0)
        colorFun.AddRGBPoint(max_npV*iRescaleSlope+iRescaleIntercept, 0.78, 0.78, 0.92, .5, 0.0)

        
        width_s=80
        opacityFun.AddPoint(min_npV*iRescaleSlope+iRescaleIntercept, 0, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
        # Right
        opacityFun.AddPoint(adjThresholds[1], 0, .5, .0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
        opacityFun.AddPoint(adjThresholds[2], 0.5, .5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
        opacityFun.AddPoint(max_npV*iRescaleSlope+iRescaleIntercept, .75, 0.5, 0.0) # IntensityValue, Opacity, Position of midpoint, sharpness of midpoint
        
        
        gradientFun.AddPoint(min_npV*iRescaleSlope+iRescaleIntercept, 1.0, 0.5,.0)
        gradientFun.AddPoint(min_npV*iRescaleSlope+iRescaleIntercept + (max_npV-min_npV)*0.2,.0,0.5,.0)
        #gradientFun.AddPoint(28.,.0,0.0,.0)
        gradientFun.AddPoint(max_npV*iRescaleSlope+iRescaleIntercept,1.0,0.5,.0)

        property.ShadeOn()
        mapper.SetBlendModeToComposite()
        property.SetAmbient(0.2)
        property.SetDiffuse(1.0)
        property.SetSpecular(0.0)
        property.SetSpecularPower(1.0)
        property.SetScalarOpacityUnitDistance(0.8919)
    volume.SetMapper(mapper)
    volume.SetProperty(property)

    return min_npV*iRescaleSlope+iRescaleIntercept,max_npV*iRescaleSlope+iRescaleIntercept, adjThresholds, [dx,dy,dz], volume