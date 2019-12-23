add_executable(TestApp ${LIBELAS_SRC_FILES} "TestApp.cpp")
target_link_libraries(TestApp ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES})
install(TARGETS TestApp DESTINATION bin)

add_executable(prop4D ${SUN_UTILS_SRC_FILES} ${CAMOT_SRC_FILES}  ${EXTERNAL_SRC_FILES} "prop4D.cpp")
target_link_libraries(prop4D ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} elas viso2 scene_segmentation connected_components tracking)
install(TARGETS prop4D DESTINATION bin)