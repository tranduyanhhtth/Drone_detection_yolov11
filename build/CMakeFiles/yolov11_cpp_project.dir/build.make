# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/danz/Downloads/Drone_detect/yolov11_cpp_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/danz/Downloads/Drone_detect/yolov11_cpp_project/build

# Include any dependencies generated for this target.
include CMakeFiles/yolov11_cpp_project.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/yolov11_cpp_project.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/yolov11_cpp_project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolov11_cpp_project.dir/flags.make

CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.o: CMakeFiles/yolov11_cpp_project.dir/flags.make
CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.o: /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/main.cpp
CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.o: CMakeFiles/yolov11_cpp_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/danz/Downloads/Drone_detect/yolov11_cpp_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.o -MF CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.o.d -o CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.o -c /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/main.cpp

CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/main.cpp > CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.i

CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/main.cpp -o CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.s

CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.o: CMakeFiles/yolov11_cpp_project.dir/flags.make
CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.o: /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/yolov11_onnx.cpp
CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.o: CMakeFiles/yolov11_cpp_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/danz/Downloads/Drone_detect/yolov11_cpp_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.o -MF CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.o.d -o CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.o -c /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/yolov11_onnx.cpp

CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/yolov11_onnx.cpp > CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.i

CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/yolov11_onnx.cpp -o CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.s

CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.o: CMakeFiles/yolov11_cpp_project.dir/flags.make
CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.o: /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/bbox.cpp
CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.o: CMakeFiles/yolov11_cpp_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/danz/Downloads/Drone_detect/yolov11_cpp_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.o -MF CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.o.d -o CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.o -c /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/bbox.cpp

CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/bbox.cpp > CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.i

CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/danz/Downloads/Drone_detect/yolov11_cpp_project/src/bbox.cpp -o CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.s

# Object files for target yolov11_cpp_project
yolov11_cpp_project_OBJECTS = \
"CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.o" \
"CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.o" \
"CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.o"

# External object files for target yolov11_cpp_project
yolov11_cpp_project_EXTERNAL_OBJECTS =

yolov11_cpp_project: CMakeFiles/yolov11_cpp_project.dir/src/main.cpp.o
yolov11_cpp_project: CMakeFiles/yolov11_cpp_project.dir/src/yolov11_onnx.cpp.o
yolov11_cpp_project: CMakeFiles/yolov11_cpp_project.dir/src/bbox.cpp.o
yolov11_cpp_project: CMakeFiles/yolov11_cpp_project.dir/build.make
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_cvv.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.6.0
yolov11_cpp_project: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.6.0
yolov11_cpp_project: CMakeFiles/yolov11_cpp_project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/danz/Downloads/Drone_detect/yolov11_cpp_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable yolov11_cpp_project"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolov11_cpp_project.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolov11_cpp_project.dir/build: yolov11_cpp_project
.PHONY : CMakeFiles/yolov11_cpp_project.dir/build

CMakeFiles/yolov11_cpp_project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolov11_cpp_project.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolov11_cpp_project.dir/clean

CMakeFiles/yolov11_cpp_project.dir/depend:
	cd /home/danz/Downloads/Drone_detect/yolov11_cpp_project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/danz/Downloads/Drone_detect/yolov11_cpp_project /home/danz/Downloads/Drone_detect/yolov11_cpp_project /home/danz/Downloads/Drone_detect/yolov11_cpp_project/build /home/danz/Downloads/Drone_detect/yolov11_cpp_project/build /home/danz/Downloads/Drone_detect/yolov11_cpp_project/build/CMakeFiles/yolov11_cpp_project.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/yolov11_cpp_project.dir/depend

