import SimpleITK as sitk
import sys
import os

def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}")

def myReg3(source, target):
    pixelType = sitk.sitkFloat32

    '''fixed = sitk.ReadImage(, sitk.sitkFloat32)'''
    fixed = sitk.Normalize(source)
    fixed = sitk.DiscreteGaussian(fixed, 2.0)

    '''moving = sitk.ReadImage(image, sitk.sitkFloat32)'''
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
    return cimg


data1A = "C:/Users/gabri/Desktop/Data/manifest-1608266677008/MIDRC-RICORD-1A/MIDRC-RICORD-1A-419639-000082/08-02-2002-NA-CT CHEST WITHOUT CONTRAST-04614/2.000000-ROUTINE CHEST NON-CON-97100"
data1B = "C:/Users/gabri/Desktop/Data/manifest-1608266677008/MIDRC-RICORD-1A/MIDRC-RICORD-1A-419639-000800/05-11-2005-NA-CT CHEST WITHOUT CONTRAST-61055/2.000000-ROUTINE CHEST NON-CON-70218"



reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(data1A)
reader.SetFileNames(dicom_names)
image1A = reader.Execute();

reader2 = sitk.ImageSeriesReader()
dicom_names = reader2.GetGDCMSeriesFileNames(data1B)
reader2.SetFileNames(dicom_names)
image1B = reader2.Execute();



regIMG = myReg3(image1A, image1B)



image_viewer1A = sitk.ImageViewer()
image_viewer1A.SetApplication('C:/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe')
image_viewer1A.Execute(image1A)

image_viewer1B = sitk.ImageViewer()
image_viewer1B.SetApplication('C:/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe')
image_viewer1B.Execute(image1B)



image_viewer = sitk.ImageViewer()
image_viewer.SetApplication('C:/Program Files/ITK-SNAP 3.8/bin/ITK-SNAP.exe')
image_viewer.Execute(regIMG)
