import SimpleITK 
import numpy as np
import json
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
#from IPython.display import clear_output
#import cv2
import itk
from skimage.util import montage
from skimage import transform
import glob
DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("/output/results.json")


# todo change with your team-name
class ThresholdModel():
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH):

        self.debug = False  # False for running the docker!
        if self.debug:
            self._input_path = Path('/path-do-input-data/')
            self._output_path = Path('/path-to-output-dir/')
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = self._output_path / 'results.json'
            self._case_results = []

        else:
            self._input_path = input_path
            self._output_path = output_path
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = DEFAULT_ALGORITHM_OUTPUT_FILE_PATH
            self._case_results = []
    
    
    def down_block(self,x, filters, kernel_size=(3, 3, 3), padding="same", strides=1):
        c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        p = keras.layers.MaxPool3D((2, 2, 2), (2, 2, 2))(c)
        return c, p
    
    def up_block(self,x, skip, filters, kernel_size=(3, 3, 3), padding="same", strides=1):
        us = keras.layers.UpSampling3D((2, 2, 2))(x)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
        c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c
    
    def bottleneck(self,x, filters, kernel_size=(3, 3, 3), padding="same", strides=1):
        c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c
    
    def UNet(self,c):
        f = [32, 64, 128, 256, 512]
        inputs = keras.layers.Input((64, 64, 64, c))
        
        p0 = inputs
        c1, p1 = self.down_block(p0, f[0]) 
        c2, p2 = self.down_block(p1, f[1]) 
        c3, p3 = self.down_block(p2, f[2]) 
        c4, p4 = self.down_block(p3, f[3]) 
        
        bn = self.bottleneck(p4, f[4])
        
        u1 = self.up_block(bn, c4, f[3]) 
        u2 = self.up_block(u1, c3, f[2]) 
        u3 = self.up_block(u2, c2, f[1]) 
        u4 = self.up_block(u3, c1, f[0]) 
        
        outputs = keras.layers.Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid")(u4)
        model = keras.models.Model(inputs, outputs)
        return model
    
    def doRegistration(self, path_fixed, path_moving):
        FixedImageType   = itk.Image[itk.F, 3]
        MovingImageType  = itk.Image[itk.F, 3]
        TransformType    = itk.VersorRigid3DTransform[itk.D]
        OptimizerType    = itk.RegularStepGradientDescentOptimizerv4[itk.D]
        RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType,
                                                     MovingImageType]
        MetricType       = itk.MeanSquaresImageToImageMetricv4[FixedImageType,
                                                           MovingImageType]
        fixedImageReader  = itk.ImageFileReader[FixedImageType].New()
        movingImageReader = itk.ImageFileReader[MovingImageType].New()
        fixedImageReader.SetFileName(path_fixed)
        movingImageReader.SetFileName(path_moving)
    
        registration = RegistrationType.New()
        registration.SetFixedImage(fixedImageReader.GetOutput())
        registration.SetMovingImage(movingImageReader.GetOutput())
        optimizer = OptimizerType.New()
        registration.SetOptimizer(optimizer)
        metric = MetricType.New()
        registration.SetMetric(metric)
        initialTransform = TransformType.New()
        TransformInitializerType = itk.CenteredTransformInitializer[
        TransformType, FixedImageType, MovingImageType];
        transformInitializer = TransformInitializerType.New()
        transformInitializer.SetTransform(initialTransform)
        transformInitializer.SetFixedImage(fixedImageReader.GetOutput())
        transformInitializer.SetMovingImage(movingImageReader.GetOutput())
        transformInitializer.MomentsOn()
        transformInitializer.InitializeTransform()
    
        axis = [0, 0, 1]
        angle = 0.05
        rotation = initialTransform.GetVersor()
        rotation.Set(axis, angle)
        initialTransform.SetRotation(rotation)
        registration.SetInitialTransform(initialTransform)
    
        numOfParam = initialTransform.GetNumberOfParameters()
        optimizerScales = itk.OptimizerParameters.D(numOfParam)
        translationScale = 1.0 / 1000.0
        optimizerScales.SetElement(0, 1.0)
        optimizerScales.SetElement(1, 1.0)
        optimizerScales.SetElement(2, 1.0)
        optimizerScales.SetElement(3, translationScale)
        optimizerScales.SetElement(4, translationScale)
        optimizerScales.SetElement(5, translationScale)
        optimizer.SetScales(optimizerScales)
    
        optimizer.SetNumberOfIterations(5000)
        optimizer.SetLearningRate(0.2)
        optimizer.SetMinimumStepLength(0.001)
        optimizer.SetReturnBestParametersAndValue(True)
    
        def cb_registration_start():
            global cb_metric_values
            global cb_current_iteration_number
            cb_metric_values = []
            cb_current_iteration_number = -1
    
        def cb_registration_end():
            global cb_metric_values
            global cb_current_iteration_number
            del cb_metric_values
            del cb_current_iteration_number
    
        def cb_registration_iteration():
            global cb_metric_values
            global cb_current_iteration_number
    
            if optimizer.GetCurrentIteration() == cb_current_iteration_number:
                return
    
            cb_current_iteration_number = optimizer.GetCurrentIteration()
            cb_metric_values.append(optimizer.GetValue())
    
        optimizer.AddObserver(itk.StartEvent(), cb_registration_start)
        optimizer.AddObserver(itk.EndEvent(), cb_registration_end)
        optimizer.AddObserver(itk.IterationEvent(), cb_registration_iteration)
    
        registration.SetNumberOfLevels(1)
        registration.SetSmoothingSigmasPerLevel([0])
        registration.SetShrinkFactorsPerLevel([1])
    
        registration.Update()
        transform = registration.GetTransform()
        finalParameters = transform.GetParameters()
        finalTransform = TransformType.New()
        finalTransform.SetFixedParameters(registration.GetOutput().Get().GetFixedParameters())
        finalTransform.SetParameters(finalParameters)
        matrix = finalTransform.GetMatrix()
        offset = finalTransform.GetOffset()
    
        ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]
        resampler = ResampleFilterType.New()
        resampler.SetInput(movingImageReader.GetOutput())
        resampler.SetTransform(finalTransform)
        resampler.SetUseReferenceImage(True)
        resampler.SetReferenceImage(fixedImageReader.GetOutput())
        resampler.SetDefaultPixelValue(0)
    
        fixedImageArray = itk.GetArrayViewFromImage(fixedImageReader.GetOutput())
        resultImageArray = itk.GetArrayViewFromImage(resampler.GetOutput())
        return resultImageArray
        del  resultImageArray,fixedImageArray,resampler,ResampleFilterType,registration,optimizer,fixedImageReader,movingImageReader,path_fixed, path_moving

    
    def pre_process(self,dwi, adc, flair):
        a=adc/np.percentile(adc[adc > 0], 90)
        #kernel=np.array([[[0,0,0],[0,1,0],[0,0, 0]],[[0, 1, 0],[1, 1, 1],[0, 1, 0]],[[0, 0, 0],[0, 1, 0],[0, 0, 0]]], dtype=np.uint8)
        #kernel=np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], dtype=np.uint8)
        #adc_mask= cv2.dilate(np.array((a<0.8), dtype=np.uint8), kernel, iterations = 1)
        adc_mask=np.array((a<0.8), dtype=np.uint8)
        d=dwi/np.percentile(dwi[dwi > 0], 99.5)
        f=flair/np.percentile(flair[flair > 1], 99.9)

        d=d*adc_mask
        a=a*adc_mask
        f=f*adc_mask
        d[d>1]=1
        a[a>1]=1
        f[f>1]=1

        return d,a,f
        del a,d,f,adc_mask,kernel,dwi, adc, flair

    def pred3d(self,dwi, adc, flair,sim_model, flag):
        new_d = transform.resize(dwi, (96, 128, 128))
        new_a = transform.resize(adc, (96, 128, 128))
        new_f = transform.resize(flair, (96, 128, 128))
        pred=np.zeros(new_d.shape)
        weightedmsk=np.zeros(new_d.shape)
        for i in range(0,new_d.shape[2]-63,32):
            for j in range(0,new_d.shape[1]-63,32):
                for k in range(0,new_d.shape[0]-63,32):
                    if flag=='1':
                        temp=np.stack((new_a[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_f[k:k+64,i:i+64,j:j+64]),-1) #ADF
                    elif flag=='2':
                        temp=np.stack((new_d[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_f[k:k+64,i:i+64,j:j+64]),-1) #DDF
                    elif flag=='3':
                        temp=np.stack((new_a[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_f[k:k+64,i:i+64,j:j+64]),-1) #ADDF
                    temp1=np.expand_dims(np.rot90(temp,axes=(1, 2)),0)
                    pred_seg = sim_model.predict(temp1/np.max(temp1),1)[0,...,0]
                    if np.max(pred_seg)!=0:
                        pred_seg=pred_seg/np.max(pred_seg)
                    weightedmsk[k:k+64,i:i+64,j:j+64]=pred[k:k+64,i:i+64,j:j+64]+np.ones(pred_seg.shape)
                    pred[k:k+64,i:i+64,j:j+64]=pred[k:k+64,i:i+64,j:j+64]+np.rot90(pred_seg,axes=(2,1))
        
        result= (transform.resize(pred/weightedmsk, dwi.shape))>0.5
           
        return  result
        del result, new_d, new_a, new_f, pred, temp, temp1, pred_seg,weightedmsk,dwi, adc, flair
        
    def predict(self, input_data):
        """
        Input   input_data, dict.
                The dictionary contains 3 images and 3 json files.
                keys:  'dwi_image' , 'adc_image', 'flair_image', 'dwi_json', 'adc_json', 'flair_json'

        Output  prediction, array.
                Binary mask encoding the lesion segmentation (0 background, 1 foreground).
        """
        # Get all image inputs.
        dwi_image, adc_image, flair_image = input_data['dwi_image'],\
                                            input_data['adc_image'],\
                                            input_data['flair_image']
                                            
        # Get all image inputs.
        dwi_p, adc_p, flair_p = input_data['dwi_path'],\
                                            input_data['adc_path'],\
                                            input_data['flair_path']        


        # Get all json inputs.
        dwi_json, adc_json, flair_json = input_data['dwi_json'],\
                                         input_data['adc_json'],\
                                         input_data['flair_json']

        ################################################################################################################
        #################################### Beginning of your prediction method. ######################################
        # todo replace with your best model here!
        # As an example, we'll segment the DWI using a 99th-percentile intensity cutoff.
        dwi_image_data=SimpleITK.GetArrayFromImage(dwi_image).astype(np.float64)
        adc_image_data=SimpleITK.GetArrayFromImage(adc_image).astype(np.float64)
        flair_image_data=self.doRegistration(dwi_p,flair_p).astype(np.float64)
        dwi_image_data, adc_image_data, flair_image_data=self.pre_process(dwi_image_data, adc_image_data, flair_image_data)
        sim_model1 = self.UNet(3)
        sim_model1.load_weights('best1.hdf5')
        sim_model2 = self.UNet(3)
        sim_model2.load_weights('best2.hdf5')
        sim_model3 = self.UNet(4)
        sim_model3.load_weights('best3.hdf5')
        prediction1=self.pred3d(dwi_image_data, adc_image_data, flair_image_data,sim_model1,'1').astype(np.uint8)
        prediction2=self.pred3d(dwi_image_data, adc_image_data, flair_image_data,sim_model2,'2').astype(np.uint8)
        prediction3=self.pred3d(dwi_image_data, adc_image_data, flair_image_data,sim_model3,'3').astype(np.uint8)
        prediction=np.zeros(prediction1.shape)
        prediction[:,:,:]=(prediction1[:,:,:]+prediction2[:,:,:]+prediction3[:,:,:])>1   

        #dwi_image_data = SimpleITK.GetArrayFromImage(dwi_image)
        #dwi_cutoff = np.percentile(dwi_image_data[dwi_image_data > 0], 99)
        #prediction = dwi_image_data > dwi_cutoff

        #################################### End of your prediction method. ############################################
        ################################################################################################################

        return prediction.astype(int)

    def process_isles_case(self, input_data, input_filename):
        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = input_data['dwi_image'].GetOrigin(),\
                                     input_data['dwi_image'].GetSpacing(),\
                                     input_data['dwi_image'].GetDirection()

        # Segment images.
        prediction = self.predict(input_data) # function you need to update!

        # Build the itk object.
        output_image = SimpleITK.GetImageFromArray(prediction)
        output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)

        # Write segmentation to output location.
        if not self._algorithm_output_path.exists():
            os.makedirs(str(self._algorithm_output_path))
        output_image_path = self._algorithm_output_path / input_filename
        SimpleITK.WriteImage(output_image, str(output_image_path))

        # Write segmentation file to json.
        if output_image_path.exists():
            json_result = {"outputs": [dict(type="Image", slug="stroke-lesion-segmentation",
                                                 filename=str(output_image_path.name))],
                           "inputs": [dict(type="Image", slug="dwi-brain-mri",
                                           filename=input_filename)]}

            self._case_results.append(json_result)
            self.save()


    def load_isles_case(self):
        """ Loads the 6 inputs of ISLES22 (3 MR images, 3 metadata json files accompanying each MR modality).
        Note: Cases missing the metadata will still have a json file, though their fields will be empty. """

        # Get MR data paths.
        dwi_image_path = self.get_file_path(slug='dwi-brain-mri', filetype='image')
        adc_image_path = self.get_file_path(slug='adc-brain-mri', filetype='image')
        flair_image_path = self.get_file_path(slug='flair-brain-mri', filetype='image')

        # Get MR metadata paths.
        dwi_json_path = self.get_file_path(slug='dwi-mri-acquisition-parameters', filetype='json')
        adc_json_path = self.get_file_path(slug='adc-mri-parameters', filetype='json')
        flair_json_path = self.get_file_path(slug='flair-mri-acquisition-parameters', filetype='json')

        input_data = {'dwi_image': SimpleITK.ReadImage(str(dwi_image_path)), 'dwi_json': json.load(open(dwi_json_path)),
                      'adc_image': SimpleITK.ReadImage(str(adc_image_path)), 'adc_json': json.load(open(adc_json_path)),
                      'flair_image': SimpleITK.ReadImage(str(flair_image_path)), 'flair_json': json.load(open(flair_json_path)),
                      'dwi_path':str(dwi_image_path),'adc_path':str(adc_image_path),'flair_path':str(flair_image_path)}

        # Set input information.
        input_filename = str(dwi_image_path).split('/')[-1]
        return input_data, input_filename

    def get_file_path(self, slug, filetype='image'):
        """ Gets the path for each MR image/json file."""

        if filetype == 'image':
            #file_list = list((self._input_path / "images" / slug).glob("*.mha"))
            file_list = list((self._input_path / "images" / slug).glob("*.mha"))
        elif filetype == 'json':
            file_list = list(self._input_path.glob("*{}.json".format(slug)))

        # Check that there is a single file to load.
        if len(file_list) != 1:
            print('Loading error')
        else:
            return file_list[0]

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results, f)

    def process(self):
        input_data, input_filename = self.load_isles_case()
        self.process_isles_case(input_data, input_filename)
        print('finished')

if __name__ == "__main__":
    # todo change with your team-name
    ThresholdModel().process()
#%%
    
import SimpleITK 
import numpy as np
import json
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
#from IPython.display import clear_output
#import cv2
import itk
from skimage.util import montage
from skimage import transform
import glob

def doRegistration(path_fixed, path_moving):
    FixedImageType   = itk.Image[itk.F, 3]
    MovingImageType  = itk.Image[itk.F, 3]
    TransformType    = itk.VersorRigid3DTransform[itk.D]
    OptimizerType    = itk.RegularStepGradientDescentOptimizerv4[itk.D]
    RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType,
                                                 MovingImageType]
    MetricType       = itk.MeanSquaresImageToImageMetricv4[FixedImageType,
                                                       MovingImageType]
    fixedImageReader  = itk.ImageFileReader[FixedImageType].New()
    movingImageReader = itk.ImageFileReader[MovingImageType].New()
    fixedImageReader.SetFileName(path_fixed)
    movingImageReader.SetFileName(path_moving)

    registration = RegistrationType.New()
    registration.SetFixedImage(fixedImageReader.GetOutput())
    registration.SetMovingImage(movingImageReader.GetOutput())
    optimizer = OptimizerType.New()
    registration.SetOptimizer(optimizer)
    metric = MetricType.New()
    registration.SetMetric(metric)
    initialTransform = TransformType.New()
    TransformInitializerType = itk.CenteredTransformInitializer[
    TransformType, FixedImageType, MovingImageType];
    transformInitializer = TransformInitializerType.New()
    transformInitializer.SetTransform(initialTransform)
    transformInitializer.SetFixedImage(fixedImageReader.GetOutput())
    transformInitializer.SetMovingImage(movingImageReader.GetOutput())
    transformInitializer.MomentsOn()
    transformInitializer.InitializeTransform()

    axis = [0, 0, 1]
    angle = 0.05
    rotation = initialTransform.GetVersor()
    rotation.Set(axis, angle)
    initialTransform.SetRotation(rotation)
    registration.SetInitialTransform(initialTransform)

    numOfParam = initialTransform.GetNumberOfParameters()
    optimizerScales = itk.OptimizerParameters.D(numOfParam)
    translationScale = 1.0 / 1000.0
    optimizerScales.SetElement(0, 1.0)
    optimizerScales.SetElement(1, 1.0)
    optimizerScales.SetElement(2, 1.0)
    optimizerScales.SetElement(3, translationScale)
    optimizerScales.SetElement(4, translationScale)
    optimizerScales.SetElement(5, translationScale)
    optimizer.SetScales(optimizerScales)

    optimizer.SetNumberOfIterations(5000)
    optimizer.SetLearningRate(0.2)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetReturnBestParametersAndValue(True)

    def cb_registration_start():
        global cb_metric_values
        global cb_current_iteration_number
        cb_metric_values = []
        cb_current_iteration_number = -1

    def cb_registration_end():
        global cb_metric_values
        global cb_current_iteration_number
        del cb_metric_values
        del cb_current_iteration_number

    def cb_registration_iteration():
        global cb_metric_values
        global cb_current_iteration_number

        if optimizer.GetCurrentIteration() == cb_current_iteration_number:
            return

        cb_current_iteration_number = optimizer.GetCurrentIteration()
        cb_metric_values.append(optimizer.GetValue())

    optimizer.AddObserver(itk.StartEvent(), cb_registration_start)
    optimizer.AddObserver(itk.EndEvent(), cb_registration_end)
    optimizer.AddObserver(itk.IterationEvent(), cb_registration_iteration)

    registration.SetNumberOfLevels(1)
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SetShrinkFactorsPerLevel([1])

    registration.Update()
    transform = registration.GetTransform()
    finalParameters = transform.GetParameters()
    finalTransform = TransformType.New()
    finalTransform.SetFixedParameters(registration.GetOutput().Get().GetFixedParameters())
    finalTransform.SetParameters(finalParameters)
    matrix = finalTransform.GetMatrix()
    offset = finalTransform.GetOffset()

    ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]
    resampler = ResampleFilterType.New()
    resampler.SetInput(movingImageReader.GetOutput())
    resampler.SetTransform(finalTransform)
    resampler.SetUseReferenceImage(True)
    resampler.SetReferenceImage(fixedImageReader.GetOutput())
    resampler.SetDefaultPixelValue(0)

    fixedImageArray = itk.GetArrayViewFromImage(fixedImageReader.GetOutput())
    resultImageArray = itk.GetArrayViewFromImage(resampler.GetOutput())
    return resultImageArray
    del  resultImageArray,fixedImageArray,resampler,ResampleFilterType,registration,optimizer,fixedImageReader,movingImageReader,path_fixed, path_moving


def pre_process(dwi, adc, flair):
    a=adc/np.percentile(adc[adc > 0], 90)
    #kernel=np.array([[[0,0,0],[0,1,0],[0,0, 0]],[[0, 1, 0],[1, 1, 1],[0, 1, 0]],[[0, 0, 0],[0, 1, 0],[0, 0, 0]]], dtype=np.uint8)
    #kernel=np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], dtype=np.uint8)
    #adc_mask= cv2.dilate(np.array((a<0.8), dtype=np.uint8), kernel, iterations = 1)
    adc_mask=np.array((a<0.8), dtype=np.uint8)
    d=dwi/np.percentile(dwi[dwi > 0], 99.5)
    f=flair/np.percentile(flair[flair > 1], 99.9)

    d=d*adc_mask
    a=a*adc_mask
    f=f*adc_mask
    d[d>1]=1
    a[a>1]=1
    f[f>1]=1

    return d,a,f
    del a,d,f,adc_mask,kernel,dwi, adc, flair

def pred3d(dwi, adc, flair, flag):
    new_d = transform.resize(dwi, (96, 128, 128))
    new_a = transform.resize(adc, (96, 128, 128))
    new_f = transform.resize(flair, (96, 128, 128))
    pred=np.zeros(new_d.shape)
    weightedmsk=np.zeros(new_d.shape)
    for i in range(0,new_d.shape[2]-63,32):
        for j in range(0,new_d.shape[1]-63,32):
            for k in range(0,new_d.shape[0]-63,32):
                if flag=='1':
                    temp=np.stack((new_a[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_f[k:k+64,i:i+64,j:j+64]),-1) #ADF
                elif flag=='2':
                    temp=np.stack((new_d[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_f[k:k+64,i:i+64,j:j+64]),-1) #DDF
                elif flag=='3':
                    temp=np.stack((new_a[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_f[k:k+64,i:i+64,j:j+64]),-1) #ADDF

       
    return  temp
    #del result, new_d, new_a, new_f, pred, temp, temp1, pred_seg,weightedmsk,dwi, adc, flair

path='C:/Users/user/Downloads/dataset-ISLES22^release1/dataset-ISLES22^release1/rawdata/sub-strokecase0003/ses-0001/sub-strokecase0003_ses-0001_dwi.nii.gz'
dwi_image_data=SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path)).astype(np.float64)
adc_image_data=SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path.replace('dwi','adc'))).astype(np.float64)
flair_image_data=doRegistration(path,path.replace('dwi','flair')).astype(np.float64)
dwi_image_data, adc_image_data, flair_image_data=pre_process(dwi_image_data, adc_image_data, flair_image_data)
#%%     
temp=SimpleITK.ReadImage(path)
origin, spacing, direction = temp.GetOrigin(),\
                                     temp.GetSpacing(),\
                                     temp.GetDirection()

output_image = SimpleITK.GetImageFromArray(flair_image_data)
output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)


SimpleITK.WriteImage(output_image, path.replace('dwi','r_flair'))

#%%
from skimage.util import montage
import matplotlib.pyplot as plt
def montage_nd(in_img):
    if len(in_img.shape)>3:
        return montage(np.stack([montage_nd(x_slice) for x_slice in in_img],0))
    elif len(in_img.shape)==3:
        return montage(in_img)
    else:
        warn('Input less than 3d image, returning original', RuntimeWarning)
        return in_img
def pred3d(dwi, adc, flair, flag):
    new_d = transform.resize(dwi, (96, 128, 128))
    new_a = transform.resize(adc, (96, 128, 128))
    new_f = transform.resize(flair, (96, 128, 128))
    pred=np.zeros(new_d.shape)
    weightedmsk=np.zeros(new_d.shape)
    for i in range(0,new_d.shape[2]-63,32):
        for j in range(0,new_d.shape[1]-63,32):
            for k in range(0,new_d.shape[0]-63,32):
                if flag=='1':
                    temp=np.stack((new_a[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_f[k:k+64,i:i+64,j:j+64]),-1) #ADF
                elif flag=='2':
                    temp=np.stack((new_d[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_f[k:k+64,i:i+64,j:j+64]),-1) #DDF
                elif flag=='3':
                    temp=np.stack((new_a[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_d[k:k+64,i:i+64,j:j+64],new_f[k:k+64,i:i+64,j:j+64]),-1) #ADDF
                if (i==32 and j==64 and k==32):
                    pred=np.array(temp)
       
    return pred

path='C:/Users/user/Downloads/dataset-ISLES22^release1/dataset-ISLES22^release1/derivatives/sub-strokecase0003/ses-0001/sub-strokecase0003_ses-0001_msk.nii.gz'
msk=SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path)).astype(np.float64)
    
aa=pred3d(msk, msk, msk, '1')
#big_ADF=np.stack((montage_nd(aa[:,...,0]),montage_nd(aa[:,...,1]),montage_nd(aa[:,...,2])),axis=-1)
#plt.imshow(big_ADF, cmap = 'bone')


plt.imshow(np.stack((aa[22,...,0],aa[22,...,1],aa[22,...,2]),axis=-1))
plt.axis('off')