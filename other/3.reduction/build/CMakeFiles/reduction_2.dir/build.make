# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake/bin/cmake

# The command to remove a file.
RM = /opt/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build

# Include any dependencies generated for this target.
include CMakeFiles/reduction_2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/reduction_2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reduction_2.dir/flags.make

CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: ../reduction_2.cu
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: ../reduction_aux.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/_G_config.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/alloca.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/assert.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/c++/5/bits/atomic_lockfree_defines.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/c++/5/bits/cpp_type_traits.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/c++/5/cmath
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/c++/5/cstdlib
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/c++/5/exception
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/c++/5/ext/type_traits.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/c++/5/new
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/endian.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/features.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/libio.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/limits.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/linux/limits.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/math.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/stdc-predef.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/stdint.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/stdio.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/stdlib.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/string.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/time.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/wchar.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/byteswap-16.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/byteswap.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/endian.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/huge_val.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/huge_valf.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/huge_vall.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/inf.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/libm-simd-decl-stubs.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/local_lim.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/math-vector.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/mathcalls.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/mathdef.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/nan.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/posix1_lim.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/posix2_lim.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/pthreadtypes.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/select.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/sigset.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/stdio_lim.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/stdlib-float.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/sys_errlist.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/time.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/timex.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/typesizes.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/waitflags.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/waitstatus.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/wchar.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/wordsize.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/bits/xopen_lim.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/c++config.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/cpu_defines.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/c++/5/bits/os_defines.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/gnu/stubs-64.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/gnu/stubs.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/sys/cdefs.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/sys/select.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/sys/sysmacros.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/sys/time.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/x86_64-linux-gnu/sys/types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/include/xlocale.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/lib/gcc/x86_64-linux-gnu/5/include-fixed/limits.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/lib/gcc/x86_64-linux-gnu/5/include-fixed/syslimits.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/lib/gcc/x86_64-linux-gnu/5/include/stdarg.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/lib/gcc/x86_64-linux-gnu/5/include/stdint.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/builtin_types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/channel_descriptor.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/common_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/device_double_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/device_double_functions.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/device_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/device_functions.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/host_config.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/host_defines.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/math_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/math_functions.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/sm_70_rt.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/crt/sm_70_rt.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/cuda.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/cuda_device_runtime_api.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/cuda_runtime.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/cuda_runtime_api.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/cuda_surface_types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/cuda_texture_types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/device_atomic_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/device_atomic_functions.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/device_launch_parameters.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/device_types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/driver_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/driver_types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/library_types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_20_atomic_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_20_atomic_functions.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_20_intrinsics.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_20_intrinsics.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_30_intrinsics.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_30_intrinsics.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_32_atomic_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_32_atomic_functions.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_32_intrinsics.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_32_intrinsics.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_35_atomic_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_35_intrinsics.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_60_atomic_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_60_atomic_functions.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_61_intrinsics.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/sm_61_intrinsics.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/surface_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/surface_indirect_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/surface_types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/texture_fetch_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/texture_indirect_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/texture_types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/vector_functions.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/vector_functions.hpp
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: /usr/local/cuda-10.1/include/vector_types.h
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o.cmake
CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o: ../reduction_2.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o"
	cd /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build/CMakeFiles/reduction_2.dir && /opt/cmake/bin/cmake -E make_directory /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build/CMakeFiles/reduction_2.dir//.
	cd /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build/CMakeFiles/reduction_2.dir && /opt/cmake/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build/CMakeFiles/reduction_2.dir//./reduction_2_generated_reduction_2.cu.o -D generated_cubin_file:STRING=/home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build/CMakeFiles/reduction_2.dir//./reduction_2_generated_reduction_2.cu.o.cubin.txt -P /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build/CMakeFiles/reduction_2.dir//reduction_2_generated_reduction_2.cu.o.cmake

# Object files for target reduction_2
reduction_2_OBJECTS =

# External object files for target reduction_2
reduction_2_EXTERNAL_OBJECTS = \
"/home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build/CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o"

reduction_2: CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o
reduction_2: CMakeFiles/reduction_2.dir/build.make
reduction_2: /usr/local/cuda-10.1/lib64/libcudart_static.a
reduction_2: /usr/lib/x86_64-linux-gnu/librt.so
reduction_2: CMakeFiles/reduction_2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable reduction_2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduction_2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reduction_2.dir/build: reduction_2

.PHONY : CMakeFiles/reduction_2.dir/build

CMakeFiles/reduction_2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reduction_2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reduction_2.dir/clean

CMakeFiles/reduction_2.dir/depend: CMakeFiles/reduction_2.dir/reduction_2_generated_reduction_2.cu.o
	cd /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build /home/fb/Learn/tutorials/CUDA编程入门/examples/reduction/build/CMakeFiles/reduction_2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reduction_2.dir/depend

