#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"

#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkMultiResolutionPyramidImageFilter.h"

#include "itkCenteredRigid2DTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkRegularStepGradientDescentOptimizer.h"

#include "itkImageMaskSpatialObject.h"

#include "itkLinearInterpolateImageFunction.h"

#include "itkRatioVarianceImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"

#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkNumericTraits.h"



namespace {
    // for reading moving and fixed images
    const unsigned int                                              Dimension = 2;
    typedef float                                                   InternalPixelType;
    typedef itk::Image<InternalPixelType, Dimension>                InternalImageType;
    
    typedef itk::ImageFileReader<InternalImageType>                 ImageReaderType;


    // for storing scaling information about moving and fixed images to be
    // re-applied after filters remove it
    typedef itk::MetaDataDictionary                                 DictionaryType;


    // for writing resampled registered images out
    typedef unsigned short                                          OutputPixelType;
    typedef itk::Image<OutputPixelType, Dimension>                  OutputImageType;

    typedef itk::ImageFileWriter<OutputImageType>                   ImageWriterType;
    

    // for casting internal image format to output format.
    typedef itk::CastImageFilter<InternalImageType,
                                 OutputImageType>                   InternalToOutputCastFilterType;
                                 
                                 
    
    // for storing masking information with which to create ImageMaskSpatialObjects
    // that can be used during registration for masking
    typedef signed short                                            MaskPixelType;
    typedef itk::Image<MaskPixelType, Dimension>                    MaskImageType;

    typedef itk::ImageFileWriter<MaskImageType>                     MaskWriterType;

    typedef itk::BinaryThresholdImageFilter<InternalImageType,
                                            MaskImageType>          BinaryThresholdToMaskFilterType;



    // registration scheme: multiple resolution
    typedef itk::MultiResolutionImageRegistrationMethod<
                                            InternalImageType,
                                            InternalImageType>      RegistrationType;
    typedef RegistrationType*                                       RegistrationPointer;


    typedef itk::MultiResolutionPyramidImageFilter<
                                    InternalImageType,
                                    InternalImageType >             ImagePyramidType;



    // transformation to use during registration: Rigid 2D translation + rotation with centred rotation  
    typedef itk::CenteredRigid2DTransform<double>                   TransformType;


    // initializer for transform: where to start registration / initial estimate
    typedef itk::CenteredTransformInitializer<TransformType,
                                              InternalImageType,
                                              InternalImageType>    TransformInitializerType;
                                              
    
    // registration cost transform optimizer
    typedef itk::RegularStepGradientDescentOptimizer                OptimizerType;

    typedef OptimizerType*                                          OptimizerPointer;
    typedef const OptimizerType*                                    ConstOptimizerPointer;
    typedef OptimizerType::ScalesType                               OptimizerScalesType;



    // registration cost function.  swap typedefs to test with alternative, established metric.
    typedef itk::RatioVarianceImageToImageMetric<InternalImageType,
    //                                             InternalImageType> MetricType;

    //typedef itk::MeanSquaresImageToImageMetric<InternalImageType,
                                               InternalImageType>   MetricType;


    // registration mask to limit region where cost function is determined
    typedef itk::ImageMaskSpatialObject<Dimension>                  MaskType;


    typedef itk::LinearInterpolateImageFunction<InternalImageType,
                                                double>             InterpolatorType;


    typedef itk::ResampleImageFilter<InternalImageType,
                                     InternalImageType>             ResampleFilterType;




    template <typename TRegistration>
    class RegistrationInterfaceCommand : public itk::Command 
    {
    public:
        typedef  RegistrationInterfaceCommand   Self;
        typedef  itk::Command                   Superclass;
        typedef  itk::SmartPointer<Self>        Pointer;
        itkNewMacro( Self );
    
    protected:
        RegistrationInterfaceCommand() {};

    public:
        typedef   TRegistration                             RegistrationType;
        typedef   RegistrationType*                         RegistrationPointer;
        typedef   itk::RegularStepGradientDescentOptimizer  OptimizerType;
        typedef   OptimizerType*                            OptimizerPointer;
        void Execute(itk::Object* object, const itk::EventObject& event)
        {
            // verify event invoked is of the right type.
            if( !(itk::IterationEvent().CheckEvent( &event )) )
            {
                return; // If not, we return without any further action.
            }

            // We then convert the input object pointer to a RegistrationPointer.
            RegistrationPointer registration = dynamic_cast<RegistrationPointer>( object );

            // If this is the first resolution level we set the maximum step length
            // (representing the first step size) and the minimum step length (representing
            // the convergence criterion) to large values.  At each subsequent resolution
            // level, we will reduce the minimum step length by a factor of 10 in order to
            // allow the optimizer to focus on progressively smaller regions. The maximum
            // step length is set up to the current step length. In this way, when the
            // optimizer is reinitialized at the beginning of the registration process for
            // the next level, the step length will simply start with the last value used
            // for the previous level. This will guarantee the continuity of the path
            // taken by the optimizer through the parameter space.

            OptimizerPointer optimizer = dynamic_cast< OptimizerPointer >( registration->GetOptimizer() );

            std::cout << "-------------------------------------" << std::endl;
            std::cout << "MultiResolution Level : "
                      << registration->GetCurrentLevel()  << std::endl;
            std::cout << std::endl;

            if ( registration->GetCurrentLevel() == 0 )
            {
                optimizer->SetMaximumStepLength( 1.00 );  
                optimizer->SetMinimumStepLength( 0.01 );
            }
            else
            {
                optimizer->SetMaximumStepLength( optimizer->GetMaximumStepLength() / 4.0 );
                optimizer->SetMinimumStepLength( optimizer->GetMinimumStepLength() / 10.0 );
            }
        }

        // Another version of the \code{Execute()} method accepting a \code{const}
        // input object is also required since this method is defined as pure virtual
        // in the base class.  This version simply returns without taking any action.
        void Execute(const itk::Object * , const itk::EventObject & )
        { return; }
    };

    typedef RegistrationInterfaceCommand<RegistrationType> RegistrationCommandType;


    // An observer that will monitor the evolution of the registration process.
    class CommandIterationUpdate : public itk::Command {
    public:
        typedef CommandIterationUpdate                              Self;
        typedef itk::Command                                        Superclass;
        typedef itk::SmartPointer<Self>                             Pointer;
        itkNewMacro( Self );
    
    protected:
        CommandIterationUpdate() {};
    
    public:
        void Execute(itk::Object* caller, const itk::EventObject& event) {
            Execute( (const itk::Object*)caller, event );
        }
    
        void Execute(const itk::Object* object, const itk::EventObject& event) {
            ConstOptimizerPointer optimizer = dynamic_cast<ConstOptimizerPointer>(object);
    
            if (!itk::IterationEvent().CheckEvent(&event))
                return;
    
            std::cout << optimizer->GetCurrentIteration() << " = ";
            std::cout << optimizer->GetValue() << " : ";
            std::cout << optimizer->GetCurrentPosition() << std::endl;
        }
    };
}


///////////
// main
///////////
int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Missing Parameters " << std::endl;
        std::cerr << "Usage: " << argv[0] << " fixedImageFile movingImageFile outputImagefile" << std::endl;
        return EXIT_FAILURE;
    }


    // read fixed and moving images
    
    ImageReaderType::Pointer fixedImageReader = ImageReaderType::New();
    fixedImageReader->SetFileName(argv[1]);
    try {
        fixedImageReader->Update();
    } catch (itk::ExceptionObject & err) {
        std::cerr << "ExceptionObject caught while reading fixed image" << std::endl; 
        std::cerr << err << std::endl; 
        return EXIT_FAILURE;
    }
    InternalImageType::Pointer fixedImage = fixedImageReader->GetOutput();
    std::cout << "reading fixed image file: " << fixedImageReader->GetFileName() << std::endl;


    ImageReaderType::Pointer movingImageReader = ImageReaderType::New();
    movingImageReader->SetFileName(argv[2]);
    try {
        movingImageReader->Update();
    } catch (itk::ExceptionObject & err) {
        std::cerr << "ExceptionObject caught while reading moving image" << std::endl; 
        std::cerr << err << std::endl; 
        return EXIT_FAILURE;
    }
    InternalImageType::Pointer movingImage = movingImageReader->GetOutput();
    std::cout << "reading moving image file: " << movingImageReader->GetFileName() << std::endl;


    std::cout << "will write result to file: " << argv[3] << std::endl;
    

    InterpolatorType::Pointer interpolator = InterpolatorType::New();


    // set up registration
    RegistrationType::Pointer registration = RegistrationType::New();
    registration->SetInterpolator(interpolator);
    registration->SetFixedImage(  fixedImage);
    registration->SetMovingImage( movingImage);


    // multiresolution image pyramids
    ImagePyramidType::Pointer fixedImagePyramid = ImagePyramidType::New();
    ImagePyramidType::Pointer movingImagePyramid = ImagePyramidType::New();
    
    registration->SetFixedImagePyramid( fixedImagePyramid );
    registration->SetMovingImagePyramid( movingImagePyramid );

    
    // registration metric
    MetricType::Pointer metric = MetricType::New();
    metric->SetInterpolator(interpolator);


    registration->SetMetric(metric);


    // initialize transform and specify to registration
    TransformType::Pointer transform = TransformType::New();
    transform->SetIdentity();

    TransformInitializerType::Pointer initializer = TransformInitializerType::New();
    initializer->SetTransform(transform);
    initializer->SetFixedImage(fixedImage);
    initializer->SetMovingImage(movingImage);
    initializer->MomentsOn();   // use centre of mass to initialize transform
    
    std::cout << "initializing transform" << std::endl;
    initializer->InitializeTransform();
    
    registration->SetTransform(transform);


    OptimizerType::Pointer optimizer = OptimizerType::New();
    registration->SetOptimizer(optimizer);


    registration->SetFixedImageRegion(fixedImage->GetBufferedRegion());
    registration->SetInitialTransformParameters(transform->GetParameters());
 
 
    OptimizerScalesType optimizerScales(transform->GetNumberOfParameters());
    const double translationScale = 1.0/1000.0;

    optimizerScales[0] = 1.0;
    optimizerScales[1] = translationScale;
    optimizerScales[2] = translationScale;
    optimizerScales[3] = translationScale;
    optimizerScales[4] = translationScale;
    
    optimizer->SetScales(optimizerScales);
    optimizer->SetMaximumStepLength(16.00);  
    optimizer->SetMinimumStepLength(0.01);
    optimizer->SetNumberOfIterations(32);


    // Connect observers for iterations and levels
    CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);

  
    RegistrationCommandType::Pointer command = RegistrationCommandType::New();
    registration->AddObserver( itk::IterationEvent(), command );
    registration->SetNumberOfLevels( 6 );



    std::cout << "Starting Registration" << std::endl;


    try {
        registration->StartRegistration(); 
    } catch( itk::ExceptionObject & err ) {
        std::cerr << "ExceptionObject caught !" << std::endl; 
        std::cerr << err << std::endl; 
        return EXIT_FAILURE;
    }


    // output registration transform results

    OptimizerType::ParametersType finalParameters = registration->GetLastTransformParameters();

    const double Rotation =          finalParameters[0];
    const double CentreX =           finalParameters[1];
    const double CentreY =           finalParameters[2];
    const double TranslationAlongX = finalParameters[3];
    const double TranslationAlongY = finalParameters[4];

    const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
    const double bestValue = optimizer->GetValue();

    std::cout << "Result = "                                 << std::endl;
    std::cout << " Rotation      = " <<  Rotation            << std::endl;
    std::cout << " Centre X      = " <<  CentreX             << std::endl;
    std::cout << " Centre Y      = " <<  CentreY             << std::endl;
    std::cout << " Translation X = " <<  TranslationAlongX   << std::endl;
    std::cout << " Translation Y = " <<  TranslationAlongY   << std::endl;
    std::cout << " Iterations    = " <<  numberOfIterations  << std::endl;
    std::cout << " Metric value  = " <<  bestValue           << std::endl;


    // resample input image
    ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
    resampleFilter->SetInput(            movingImage);
    resampleFilter->SetInterpolator(     interpolator);

    resampleFilter->SetTransform(        registration->GetOutput()->Get());

    resampleFilter->SetSize(             fixedImage->GetLargestPossibleRegion().GetSize());
    resampleFilter->SetOutputOrigin(     fixedImage->GetOrigin());
    resampleFilter->SetOutputSpacing(    fixedImage->GetSpacing());
    resampleFilter->SetOutputDirection(  fixedImage->GetDirection());
    resampleFilter->SetDefaultPixelValue(0);



    // cast resampled image to output type and apply meta data
    InternalToOutputCastFilterType::Pointer castFilter = InternalToOutputCastFilterType::New();
    castFilter->SetInput(resampleFilter->GetOutput());


    // get resampled and casted output image
    OutputImageType::Pointer outputImage = castFilter->GetOutput();

    // get original input moving image meta data
    const DictionaryType& movingImageMetaData = movingImage->GetMetaDataDictionary();
    
    // copy meta data from original input moving image to output image
    outputImage->SetMetaDataDictionary(movingImageMetaData);


    // output resampled image
    ImageWriterType::Pointer writer = ImageWriterType::New();

    writer->SetInput(outputImage);
    writer->SetFileName(argv[3]);
    writer->Update();


    return EXIT_SUCCESS;
}

