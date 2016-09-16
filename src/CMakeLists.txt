
include_directories(
  basic
)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()

add_subdirectory(NNCRFTagger)
add_subdirectory(NNSEmbSemiCRFTagger)
add_subdirectory(NNSEmbSemiO1CRFTagger)
add_subdirectory(NNSemiCRFTagger)
add_subdirectory(NNSemiO1CRFTagger)


