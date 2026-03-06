# core
set(CORE_HEADERS 
    "${CMAKE_SOURCE_DIR}/include/core/Value.h"
    "${CMAKE_SOURCE_DIR}/include/core/Node.h"
    "${CMAKE_SOURCE_DIR}/include/core/Tape.h"
)
set(CORE_SOURCES
    core/Value.cpp
    core/Tape.cpp
)
add_library(core
    ${CORE_SOURCES} 
    ${CORE_HEADERS}
)
target_include_directories(core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

# nn
set(NN_HEADERS
    "${CMAKE_SOURCE_DIR}/include/nn/layers/Layer.h"
    "${CMAKE_SOURCE_DIR}/include/nn/layers/Linear.h"
    "${CMAKE_SOURCE_DIR}/include/nn/layers/ReLU.h"
    "${CMAKE_SOURCE_DIR}/include/nn/layers/Tanh.h"

    "${CMAKE_SOURCE_DIR}/include/nn/models/MLP.h"

    "${CMAKE_SOURCE_DIR}/include/nn/optimizers/Optimizer.h"
    "${CMAKE_SOURCE_DIR}/include/nn/optimizers/SGD.h"
)
set(NN_SOURCES
    nn/layers/Linear.cpp
    nn/layers/ReLU.cpp
    nn/layers/Tanh.cpp

    nn/models/MLP.cpp

    nn/optimizers/SGD.cpp
)
add_library(nn
    ${NN_SOURCES} 
    ${NN_HEADERS}
)
target_include_directories(nn PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)