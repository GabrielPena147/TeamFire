
import SimpleITK as sitk

'''Data1A is some random healthy lung used to compare the data in data1B'''
'''Data1A should not be changed only Data1B'''
data1A = "./Data/manifest-1608266677008/MIDRC-RICORD-1A/MIDRC-RICORD-1A-419639-000082/08-02-2002-NA-CT CHEST WITHOUT CONTRAST-04614/2.000000-ROUTINE CHEST NON-CON-97100"

'''Data1B can be a healthy or covid lung and is what we are testing'''

'''data1B = "C:/Users/gabri/Desktop/Data/manifest-1608266677008/MIDRC-RICORD-1A\MIDRC-RICORD-1A-419639-000800/05-11-2005-NA-CT CHEST WITHOUT CONTRAST-61055/2.000000-ROUTINE CHEST NON-CON-70218"'''
'''data1B = "C:/Users/gabri/Desktop/Data/manifest-1608266677008/MIDRC-RICORD-1A/MIDRC-RICORD-1A-419639-000361/10-21-2002-NA-CT CHEST WITHOUT CONTRAST-91670/2.000000-ROUTINE CHEST NON-CON-50599"'''
'''data1B = "C:/Users/gabri/Desktop/Data/manifest-1612365584013/MIDRC-RICORD-1B/MIDRC-RICORD-1B-419639-000350/02-09-2006-NA-CT CHEST WITHOUT CONTRAST-03488/2.000000-ROUTINE CHEST NON-CON-95169"'''
data1B = "C:/Users/gabri/Desktop/Data/manifest-1612365584013/MIDRC-RICORD-1B/MIDRC-RICORD-1B-419639-000425/05-28-2002-NA-CT CHEST WITHOUT CONTRAST-88172/2.000000-ROUTINE CHEST NON-CON-50407"

def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}")

"""Transform based Registration"""
'''Bspline too long for runtime'''
def myReg3(source, target):
    pixelType = sitk.sitkFloat32

    
    fixed = sitk.Normalize(source)
    fixed = sitk.DiscreteGaussian(fixed, 2.0)

    
    moving = sitk.Normalize(target)
    moving = sitk.DiscreteGaussian(moving, 2.0)

    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsJointHistogramMutualInformation()

    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
                                              numberOfIterations=200,
                                              convergenceMinimumValue=1e-5,
                                              convergenceWindowSize=5)

    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    return cimg, R.GetMetricValue()





reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(data1A)
reader.SetFileNames(dicom_names)
image1A = reader.Execute();

reader2 = sitk.ImageSeriesReader()
dicom_names = reader2.GetGDCMSeriesFileNames(data1B)
reader2.SetFileNames(dicom_names)
image1B = reader2.Execute();


feature_img = sitk.GradientMagnitude(image1A)
feature_img2 = sitk.GradientMagnitude(image1B)

'''Watershed segmentation'''
'''Change the level value'''
ws_img1 = sitk.MorphologicalWatershed(feature_img, level=60, markWatershedLine=True, fullyConnected=False)
ws_img2 = sitk.MorphologicalWatershed(feature_img2, level=60, markWatershedLine=True, fullyConnected=False)

'''Registration function returns an image and the metric value'''
regIMG, mVal = myReg3(ws_img1, ws_img2)


image_viewer1A = sitk.ImageViewer()
image_viewer1A.SetApplication('C:/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe')
image_viewer1A.Execute(ws_img1)


image_viewer1B = sitk.ImageViewer()
image_viewer1B.SetApplication('C:/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe')
image_viewer1B.Execute(ws_img2)


image_viewer = sitk.ImageViewer()
image_viewer.SetApplication('C:/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe')
image_viewer.Execute(regIMG)