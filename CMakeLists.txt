INCLUDE(${CMAKE_SOURCE_DIR}/IJMacros.txt)

PROJECT(ImageRegTest)


ENABLE_TESTING()
INCLUDE(Dart)


#External dependencies
SET(Required_Packages
ITK
)
FOREACH(Package ${Required_Packages})
  LOADPACKAGE(${Package})
ENDFOREACH(Package)


#library dependencies
SET(Libraries
ITKCommon
ITKIO
ITKNumerics
)


#executable
SET(CurrentExe "RatioVarianceTest")
ADD_EXECUTABLE(${CurrentExe} Reg2dMultiresTest.cxx) 
TARGET_LINK_LIBRARIES(${CurrentExe} ${Libraries})
