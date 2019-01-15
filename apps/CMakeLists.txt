add_executable(TestApp ${LIBELAS_SRC_FILES} "TestApp.cpp")
target_link_libraries(TestApp ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES})
install(TARGETS TestApp DESTINATION bin)

add_executable(CAMOT ${SUN_UTILS_SRC_FILES} ${CAMOT_SRC_FILES}  ${EXTERNAL_SRC_FILES} "CAMOT.cpp")
target_link_libraries(CAMOT ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} elas viso2 scene_segmentation connected_components tracking)
install(TARGETS CAMOT DESTINATION bin)