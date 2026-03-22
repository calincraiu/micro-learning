add_executable(runtime 
    main.cpp
)

# Define the project root directory as a compile-time constant
target_compile_definitions(runtime PRIVATE
    PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

# Link the app to libraries
target_link_libraries(runtime PRIVATE
    core
    nn
    math
    util
)