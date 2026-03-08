cmake_minimum_required(VERSION 3.23)
project(micro-learning VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src) # Libs
add_subdirectory(app) # App