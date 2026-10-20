# Author: Shashwat Sharma

# Copyright 2021 Shashwat Sharma and Piero Triverio

# This file is part of Strata.

# Strata is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Strata is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Strata.  If not, see <https://www.gnu.org/licenses/>.

# set(STRATA_LIB"${CMAKE_CURRENT_SOURCE_DIR}/../build/${CMAKE_SHARED_LIBRARY_PREFIX}strata${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(STRATA_INC "${CMAKE_CURRENT_SOURCE_DIR}/../inc")

SET(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../inc)
SET(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ------ Executables ------

add_executable(dev_sandbox ${SRC_DIR}/dev_sandbox.cpp)
target_include_directories(dev_sandbox PRIVATE ${INC_DIR} ${STRATA_INC})
target_link_libraries(dev_sandbox PRIVATE strata)

add_executable(testMGF ${SRC_DIR}/testMGF.cpp)
target_include_directories(testMGF PRIVATE ${INC_DIR} ${STRATA_INC})
target_link_libraries(testMGF PRIVATE strata)

add_executable(testInterp ${SRC_DIR}/testInterp.cpp)
target_include_directories(testInterp PRIVATE ${INC_DIR} ${STRATA_INC})
target_link_libraries(testInterp PRIVATE strata)

add_executable(testDCIM ${SRC_DIR}/testDCIM.cpp)
target_include_directories(testDCIM PRIVATE ${INC_DIR} ${STRATA_INC})
target_link_libraries(testDCIM PRIVATE strata)

add_executable(testSingularity ${SRC_DIR}/testSingularity.cpp)
target_include_directories(testSingularity PRIVATE ${INC_DIR} ${STRATA_INC})
target_link_libraries(testSingularity PRIVATE strata)

add_executable(testSpectralMGF ${SRC_DIR}/testSpectralMGF.cpp)
target_include_directories(testSpectralMGF PRIVATE ${INC_DIR} ${STRATA_INC})
target_link_libraries(testSpectralMGF PRIVATE strata)

add_executable(testDjordjevicSarkar ${SRC_DIR}/testDjordjevicSarkar.cpp)
target_include_directories(testDjordjevicSarkar PRIVATE ${INC_DIR} ${STRATA_INC})
target_link_libraries(testDjordjevicSarkar PRIVATE strata)


# ------ Audit test suite (test/audit) ------

add_executable(strata_probe ${SRC_DIR}/audit/strata_probe.cpp)
target_include_directories(strata_probe PRIVATE ${INC_DIR} ${STRATA_INC} ${BOOST_INC} ${AmosBessel_INC} ${YAML_INC})
target_link_libraries(strata_probe PRIVATE strata ${AmosBessel_LIB} ${YAML_LIB})

add_executable(strata_unit_tests ${SRC_DIR}/audit/unit_tests.cpp)
target_include_directories(strata_unit_tests PRIVATE ${INC_DIR} ${STRATA_INC} ${BOOST_INC} ${AmosBessel_INC})
target_link_libraries(strata_unit_tests PRIVATE strata ${AmosBessel_LIB})
