
include_directories(
  basic
)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()

add_executable(NNCRFTagger NNCRFTagger.cpp)
add_executable(NNSemiCRFTagger NNSemiCRFTagger.cpp)
add_executable(NNSEmbSemiCRFTagger NNSEmbSemiCRFTagger.cpp)
add_executable(NNSemiO1CRFTagger NNSemiO1CRFTagger.cpp)
add_executable(NNSEmbSemiO1CRFTagger NNSEmbSemiO1CRFTagger.cpp)

