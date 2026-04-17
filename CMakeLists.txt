cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerNewton)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/")
set(EXTENSION_CATEGORY "Simulation")
set(EXTENSION_CONTRIBUTORS "Piotr Korzeniowski (NVIDIA)")
set(EXTENSION_DESCRIPTION "Run Newton Physics (GPU-accelerated, built on NVIDIA Warp) inside 3D Slicer. Drop rigid and soft bodies into the MRML scene and simulate in real time.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(NewtonPhysics)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
