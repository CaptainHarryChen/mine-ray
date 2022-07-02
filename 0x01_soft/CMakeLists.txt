find_package(OpenGL REQUIRED)

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(0x01_soft
  ${embedded_ptx_code}
  main.cpp
  )

target_link_libraries(0x01_soft
  gdt

  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}

  mainWindow
  optixRenderer
  glfw
  ${OPENGL_gl_LIBRARY}
  )
