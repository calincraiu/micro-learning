add_executable(runtime 
    main.cpp
)

# Link the app to libraries
target_link_libraries(runtime PRIVATE
    core
    nn
    math
)