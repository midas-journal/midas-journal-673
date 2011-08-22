#ifndef __itkRatioVarianceImageToImageMetric_h
#define __itkRatioVarianceImageToImageMetric_h

#include "itkImageToImageMetric.h"
#include "itkCovariantVector.h"
#include "itkPoint.h"


namespace itk
{
/** \class RatioVarianceImageToImageMetric
 * \brief Computes similarity between two images to be registered
 *
 * This metric computes the variance of the ratio of the values of pixels in
 * the fixed image and pixels in the moving image. The spatial correspondance
 * between fixed and moving image is established through a Transform. Pixel
 * values are taken from the fixed image, their positions are mapped to the
 * moving image and result in general in non-grid position on it. Values at
 * these non-grid position of the moving image are interpolated using a
 * user-selected Interpolator.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT RatioVarianceImageToImageMetric : 
    public ImageToImageMetric< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef RatioVarianceImageToImageMetric                 Self;
  typedef ImageToImageMetric<TFixedImage, TMovingImage >  Superclass;
  typedef SmartPointer<Self>                              Pointer;
  typedef SmartPointer<const Self>                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(RatioVarianceImageToImageMetric, Object);

 
  /** Types transferred from the base class */
  typedef typename Superclass::RealType                 RealType;
  typedef typename Superclass::TransformType            TransformType;
  typedef typename Superclass::TransformPointer         TransformPointer;
  typedef typename Superclass::TransformParametersType  TransformParametersType;
  typedef typename Superclass::TransformJacobianType    TransformJacobianType;
  typedef typename Superclass::GradientPixelType        GradientPixelType;
  typedef typename Superclass::OutputPointType          OutputPointType;
  typedef typename Superclass::InputPointType           InputPointType;

  typedef typename Superclass::MeasureType              MeasureType;
  typedef typename Superclass::DerivativeType           DerivativeType;
  typedef typename Superclass::FixedImageType           FixedImageType;
  typedef typename Superclass::MovingImageType          MovingImageType;
  typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer  MovingImageConstPointer;


  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType& parameters,
                      DerivativeType& Derivative ) const;

  /** Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType& parameters ) const;

  /** Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType& parameters,
                              MeasureType& Value, DerivativeType& Derivative ) const;

protected:
  RatioVarianceImageToImageMetric();
  virtual ~RatioVarianceImageToImageMetric() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  RatioVarianceImageToImageMetric(const Self&); // purposely not implemented
  void operator=(const Self&);                  // purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRatioVarianceImageToImageMetric.txx"
#endif

#endif
