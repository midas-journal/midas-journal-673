#ifndef __itkRatioVarianceImageToImageMetric_txx
#define __itkRatioVarianceImageToImageMetric_txx

#include "itkRatioVarianceImageToImageMetric.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

/**
 * Constructor
 */
template <class TFixedImage, class TMovingImage> 
RatioVarianceImageToImageMetric< TFixedImage, TMovingImage >
::RatioVarianceImageToImageMetric()
{
}


/**
 * Get the match Measure
 */
template <class TFixedImage, class TMovingImage> 
typename RatioVarianceImageToImageMetric< TFixedImage, TMovingImage >::MeasureType
RatioVarianceImageToImageMetric< TFixedImage, TMovingImage >
::GetValue( const TransformParametersType& parameters ) const
{
    // if moving image voxel values are m(i) and fixed image f(i) and the ratio
    // x(i) = m(i) / f(i), then the registration metric being calculated is
    // the variance of all x(i).  The variance of a sample of n values can be
    // estimated using
    //    M = (1 / (n - 1)) * ( sum(x(i)*x(i)) - (1/n)*sum(x(i))*sum(x(i)) )
    // ratios_sum = sum(x(i)) and ratios2_sum = sum(x(i)*x(i))
    // where i is an index of fixed image voxels

    itkDebugMacro("GetDerivative( " << parameters << " ) ");


    FixedImageConstPointer fixedImage = this->m_FixedImage;
    if ( !fixedImage ) 
    {
        itkExceptionMacro( << "Fixed image has not been assigned" );
    }


    typedef itk::ImageRegionConstIteratorWithIndex<FixedImageType> FixedIteratorType;
    FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );


    typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
    AccumulateType ratios_sum = NumericTraits< AccumulateType >::Zero;
    AccumulateType ratios2_sum = NumericTraits< AccumulateType >::Zero;
  

    this->m_NumberOfPixelsCounted = 0;
    this->SetTransformParameters( parameters );


    // loop through all pixels in fixed image, finding ratio of pixel values
    // at each, and adding to accumulators
    ti.GoToBegin();
    while ( !ti.IsAtEnd() )
    {
        typename FixedImageType::IndexType index = ti.GetIndex();
    
        // get physical point in fixed image corresponding to index
        InputPointType inputPoint;
        fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );


        // reject points outside fixed image mask, if mask is present
        if ( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
        {
            ++ti;
            continue;
        }
        
        
        // reject points which have a fixed image pixel value of zero, to
        // prevent divide by zero problems when calculating ratios
        const RealType fixedValue = ti.Get();
        if ( fixedValue == NumericTraits< RealType >::Zero )
        {
            ++ti;
            continue;
        }


        // get physical point in moving image correspnoding to index using transform
        OutputPointType transformedPoint = this->m_Transform->TransformPoint( inputPoint );


        // reject points outside moving image mask, if mask is present
        if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
        {
            ++ti;
            continue;
        }

        
        // if point is in masks and inside interpolator buffer, add the ratio of
        // pixel values to accumulators
        if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
        {
            // get moving value and calculate ratio between moving and fixed
            // fixed values at current index
            const RealType movingValue = this->m_Interpolator->Evaluate( transformedPoint );
            const RealType ratio = movingValue / fixedValue;
            
            // add ratio and ratio squared to accumulators
            ratios_sum += ratio;
            ratios2_sum += ratio*ratio;
            
            this->m_NumberOfPixelsCounted++;
        }
        ++ti;
    }
    

    const unsigned long& n = this->m_NumberOfPixelsCounted;

    
    // abort if no data with which to calculate variance
    if( !n )
    {
        itkExceptionMacro( << "All the points mapped to outside of the moving image");
    }
    
    
    // use intermediate sums to calculate variance of ratios of pixel values.
    if ( n > 1 )
    {
        // simple variance calculation for sample of data
        MeasureType measure = (ratios2_sum - (ratios_sum * ratios_sum / n)) / (n - 1.0);
        return measure;
    }

    // default if only one pixel value ratio found
    return NumericTraits< MeasureType >::Zero;
}


/**
 * Get the Derivative Measure
 */
template < class TFixedImage, class TMovingImage> 
void
RatioVarianceImageToImageMetric< TFixedImage, TMovingImage >
::GetDerivative( const TransformParametersType& parameters,
                 DerivativeType& derivative ) const
{
    // (as above...)
    // if moving image voxel values are m(i) and fixed image f(i) and the ratio
    //    x(i) = m(i) / f(i)
    // then the registration metric being calculated is the variance of all
    // x(i).  The variance of a sample of n values can be estimated using
    //    M = (1 / (n - 1)) * ( sum(x(i)*x(i)) - (1/n)*sum(x(i))*sum(x(i)) )
    // where
    //    ratios_sum = sum(x(i)) and ratios2_sum = sum(x(i)*x(i))
    // where i is an index of fixed image voxels
    //
    // the derivative of M with respect to a transformation parameter T is:
    //    dM/dT = (2 / (n - 1)) * ( sum(x(i)*dx(i)/dT) - (1/n)*sum(x(i))*sum(dx(i)/dT) )
    // where, because the moving image changes with the transform,
    //    dx(i)/dT  =  (d/dT)( m(i) / f(i) )  =  (1/f(i)) * dm(i)/dT
    //
    // if
    //    f(i) = f(p(i))
    // where p(i) is the physical point corresponding to the point in the fixed
    // image f with index i, and
    //    q(i) = S(p(i), T) 
    // where q(i) is the physical point in the moving image that the transform
    // S maps physical point p when the transform parameter T is specified, and
    //    m(i) = m(q(i)) = m(S(p(i), T))
    // then
    // dm(i)/dT = (d/dT)(m(i)) = (d/dT)(m(S(p(i), T)))
    //          = (dm(S(p(i), T))/dS) * (dS(p(i), T)/dT)
    //          = (dm(q(i))/dq) * (dS(p(i), T)/dT)
    // where dm(q(i))/dq is the derivative of the moving image voxel value as
    // the physical point in that image that is being sampled changes, or the
    // gradient of m, and
    // where dS(p(i), T)/dT is the vector of how the physical point q in the
    // moving image that corresponds to the fixed image physical point p is
    // changing as the transform parameter T is changed, or the Jacobian of the
    // transform S
    //
    // so,
    //    dx(i)/dT = (1/f(i)) * [ Grad(m) (dot) Jacobian(S) ]
    // where the Gradient of m and Jacobian component for each transform
    // parameter are evaluated for each spatial dimension of f (and m) and summed

    itkDebugMacro("GetDerivative( " << parameters << " ) ");
    
    
    if ( !this->GetGradientImage() )
    {
        itkExceptionMacro( << "The gradient image is null, maybe you forgot to call Initialize()");
    }


    FixedImageConstPointer fixedImage = this->m_FixedImage;
    if ( !fixedImage ) 
    {
        itkExceptionMacro( << "Fixed image has not been assigned" );
    }


    typedef itk::ImageRegionConstIteratorWithIndex<FixedImageType> FixedIteratorType;
    FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );


    typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
    AccumulateType ratios_sum = NumericTraits< AccumulateType >::Zero;
  

    this->m_NumberOfPixelsCounted = 0;
    this->SetTransformParameters( parameters );


    const unsigned int ParametersDimension = this->GetNumberOfParameters();
    
    derivative = DerivativeType( ParametersDimension );
    derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

    DerivativeType derivative_sums = DerivativeType( ParametersDimension );
    derivative_sums.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

    DerivativeType derivative_product_sums = DerivativeType( ParametersDimension );
    derivative_product_sums.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );



    // loop through all pixels in fixed image, finding ratio of pixel values
    // and derivatives of ratios with respect to transform parameters at each
    ti.GoToBegin();
    while ( !ti.IsAtEnd() )
    {
        typename FixedImageType::IndexType index = ti.GetIndex();
    
        // get physical point in fixed image corresponding to index
        InputPointType inputPoint;
        fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );


        // reject points outside fixed image mask, if mask is present
        if ( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
        {
            ++ti;
            continue;
        }
        
        
        // reject points which have a fixed image pixel value of zero, to
        // prevent divide by zero problems when calculating ratios
        const RealType fixedValue = ti.Get();
        if ( fixedValue == NumericTraits< RealType >::Zero )
        {
            ++ti;
            continue;
        }


        // get physical point in moving image correspnoding to index using transform
        OutputPointType transformedPoint = this->m_Transform->TransformPoint( inputPoint );


        // reject points outside moving image mask, if mask is present
        if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
        {
            ++ti;
            continue;
        }

        
        // if point is in masks and inside interpolator buffer, add the ratio of
        // pixel values to accumulators
        if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
        {
            // get moving value and calculate ratio between moving and fixed
            // fixed values at current index
            const RealType movingValue = this->m_Interpolator->Evaluate( transformedPoint );
            const RealType ratio = movingValue / fixedValue;
            
            this->m_NumberOfPixelsCounted++;
      
      
            const TransformJacobianType& jacobian = this->m_Transform->GetJacobian( inputPoint ); 
      

            // Get the gradient by NearestNeighboorInterpolation: 
            // which is equivalent to round up the point components.
            typedef typename OutputPointType::CoordRepType                          CoordRepType;
            typedef ContinuousIndex<CoordRepType, MovingImageType::ImageDimension>  MovingImageContinuousIndexType;

            MovingImageContinuousIndexType tempIndex;
            this->m_MovingImage->TransformPhysicalPointToContinuousIndex( transformedPoint, tempIndex );

            typename MovingImageType::IndexType mappedIndex; 
            mappedIndex.CopyWithRound( tempIndex );

            const GradientPixelType gradient = this->GetGradientImage()->GetPixel( mappedIndex );


            // add ratio to accumulator...
            ratios_sum += ratio;


            // for each transform parameter, find contributions to derivative
            for (unsigned int par = 0; par < ParametersDimension; par++)
            {
                for (unsigned int dim = 0; dim < FixedImageType::ImageDimension; dim++)
                {
                    const RealType ratio_derivative = jacobian( dim, par ) * gradient[dim] / fixedValue;
                    derivative_sums[par] += ratio_derivative;
                    derivative_product_sums[par] += ratio * ratio_derivative;
                }
            }
        }
        ++ti;
    }
    
    
    const unsigned long& n = this->m_NumberOfPixelsCounted;
 
   
    // abort if no data with which to calculate variance
    if ( !n )
    {
        itkExceptionMacro( << "All the points mapped to outside of the moving image");
    }
    

    // use intermediate sums to calculate derivatives of variance of ratios of
    // pixel values with respect to transformation parameters
    if ( n > 1 )
    {
        for (unsigned int par = 0; par < ParametersDimension; par++)
        {
            derivative[par] = 2.0 / (n - 1.0) * ( derivative_product_sums[par] - ((ratios_sum * derivative_sums[par]) / n) );
        }
    }

    // if n == 1, return derivatives of 0, which are the default since
    // derivative is initialized to all NumericTraits< MeasureType >::Zero
}


/*
 * Get both the match Measure and theDerivative Measure 
 */
template < class TFixedImage, class TMovingImage > 
void
RatioVarianceImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivative( const TransformParametersType& parameters, 
                         MeasureType& value, DerivativeType& derivative) const
{
    itkDebugMacro("GetDerivative( " << parameters << " ) ");
    
    
    if ( !this->GetGradientImage() )
    {
        itkExceptionMacro( << "The gradient image is null, maybe you forgot to call Initialize()");
    }


    FixedImageConstPointer fixedImage = this->m_FixedImage;
    if ( !fixedImage ) 
    {
        itkExceptionMacro( << "Fixed image has not been assigned" );
    }


    typedef itk::ImageRegionConstIteratorWithIndex<FixedImageType> FixedIteratorType;
    FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );


    typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
    AccumulateType ratios_sum = NumericTraits< AccumulateType >::Zero;
    AccumulateType ratios2_sum = NumericTraits< AccumulateType >::Zero;

    this->m_NumberOfPixelsCounted = 0;
    this->SetTransformParameters( parameters );


    const unsigned int ParametersDimension = this->GetNumberOfParameters();
    
    derivative = DerivativeType( ParametersDimension );
    derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

    DerivativeType derivative_sums = DerivativeType( ParametersDimension );
    derivative_sums.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

    DerivativeType derivative_product_sums = DerivativeType( ParametersDimension );
    derivative_product_sums.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );



    // loop through all pixels in fixed image, finding ratio of pixel values
    // and derivatives of ratios with respect to transform parameters at each
    ti.GoToBegin();
    while ( !ti.IsAtEnd() )
    {
        typename FixedImageType::IndexType index = ti.GetIndex();
    
        // get physical point in fixed image corresponding to index
        InputPointType inputPoint;
        fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );


        // reject points outside fixed image mask, if mask is present
        if ( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
        {
            ++ti;
            continue;
        }
        
        
        // reject points which have a fixed image pixel value of zero, to
        // prevent divide by zero problems when calculating ratios
        const RealType fixedValue = ti.Get();
        if ( fixedValue == NumericTraits< RealType >::Zero )
        {
            ++ti;
            continue;
        }


        // get physical point in moving image correspnoding to index using transform
        OutputPointType transformedPoint = this->m_Transform->TransformPoint( inputPoint );


        // reject points outside moving image mask, if mask is present
        if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
        {
            ++ti;
            continue;
        }

        
        // if point is in masks and inside interpolator buffer, add the ratio of
        // pixel values to accumulators
        if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
        {
            // get moving value and calculate ratio between moving and fixed
            // fixed values at current index
            const RealType movingValue = this->m_Interpolator->Evaluate( transformedPoint );
            const RealType ratio = movingValue / fixedValue;
            
            this->m_NumberOfPixelsCounted++;
      
      
            const TransformJacobianType& jacobian = this->m_Transform->GetJacobian( inputPoint ); 
      

            // Get the gradient by NearestNeighboorInterpolation: 
            // which is equivalent to round up the point components.
            typedef typename OutputPointType::CoordRepType                          CoordRepType;
            typedef ContinuousIndex<CoordRepType, MovingImageType::ImageDimension>  MovingImageContinuousIndexType;

            MovingImageContinuousIndexType tempIndex;
            this->m_MovingImage->TransformPhysicalPointToContinuousIndex( transformedPoint, tempIndex );

            typename MovingImageType::IndexType mappedIndex; 
            mappedIndex.CopyWithRound( tempIndex );

            const GradientPixelType gradient = this->GetGradientImage()->GetPixel( mappedIndex );


            // add ratio to accumulator...
            ratios_sum += ratio;
            ratios2_sum += ratio*ratio;


            // for each transform parameter, find contributions to derivative
            for (unsigned int par = 0; par < ParametersDimension; par++)
            {
                for (unsigned int dim = 0; dim < FixedImageType::ImageDimension; dim++)
                {
                    const RealType ratio_derivative = jacobian( dim, par ) * gradient[dim] / fixedValue;
                    derivative_sums[par] += ratio_derivative;
                    derivative_product_sums[par] += ratio * ratio_derivative;
                }
            }
        }
        ++ti;
    }
    
    
    const unsigned long& n = this->m_NumberOfPixelsCounted;
    

    // abort if no data with which to calculate variance
    if ( !n )
    {
        itkExceptionMacro( << "All the points mapped to outside of the moving image");
    }
    

    // use intermediate sums to calculate derivatives of variance of ratios of
    // pixel values with respect to transformation parameters
    if ( n > 1 )
    {
        for (unsigned int par = 0; par < ParametersDimension; par++)
        {
            derivative[par] = 2.0 / (n - 1.0) * ( derivative_product_sums[par] - ((ratios_sum * derivative_sums[par]) / n) );
        }

        // simple variance calculation for sample of data
        value = (ratios2_sum - (ratios_sum * ratios_sum / n)) / (n - 1.0);
    } else {
        // leave derivatives = 0, which are the default since derivative is
        // initialized to all NumericTraits< MeasureType >::Zero
        
        value = NumericTraits< MeasureType >::Zero;        
    }
}


/**
 * PrintSelf
 */
template < class TFixedImage, class TMovingImage > 
void
RatioVarianceImageToImageMetric< TFixedImage, TMovingImage >
::PrintSelf(std::ostream& os, Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}

} // end namespace itk


#endif
