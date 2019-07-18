cmake_minimum_required (VERSION 3.9)
project (AQUAD)

find_package(OpenMP)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(aquad1 aquad1.c)
target_link_libraries(aquad1 ${MPI_LIBRARIES} m)

add_executable(aquad2 aquad2.c)
target_link_libraries(aquad2 ${MPI_LIBRARIES} m)

add_executable(aquad3 aquad3.c)
target_link_libraries(aquad3 ${MPI_LIBRARIES} m)

add_executable(aquad3-version2 aquad3-version2.c)
target_link_libraries(aquad3-version2 ${MPI_LIBRARIES} m)
