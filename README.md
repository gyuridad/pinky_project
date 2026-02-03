# controller 
  - 상위 제어의 테스크를 수행한다.
  - 테스크 목록
    - 목표 좌표까지 이동
    - aruco marker 확인 후 도킹
  - 목표좌표 이동 관련 주요 기능
    - p제어로 이동
    - 고정된 waypoints 여러개 생성하여 도로 생성
    - dijkstra 알고리즘을 활용한 최적의 경로 생성
    - 전방 장애물 발견 시 로봇 멈춤
  - 부가 기능
    - 배터리 저전압 시 대기장소로 복귀
    - aruco marker 도킹 시 map에서의 로봇 위치 보정 
### Terminal Execution
  - (핑키로봇 시동걸기) ros2 launch pinky_bringup bringup_robot.launch.xml
  - (핑키로봇 네비게이션 켜기) ros2 launch pinky_navigation bringup_launch.xml map:=empty_map.yaml
  - (핑키로봇 네비게이션 뷰 보기) ros2 launch pinky_navigation nav2_view.launch.xml
  - ros2 run sensors aruco_publisher
  - ros2 run sensors battery_publisher
  - ros3 run sensors lidar_publisher
  - ros2 run actions follow_aruco_action
  - ros2 run actions goal_mover_obs_avoid_action
  - ros2 run controller controller
  - (상위 제어에서 하달된 가상의 명령)
ros2 action send_goal /pinky1/actions/execute_mission pinky_interfaces/action/ExecuteMission \
"{plan_json: '{
  \"mission_id\":\"m_dock_test_001\",
  \"steps\":[
    {\"task\":\"move_to\",\"x\":0.450,\"y\":0.424,\"yaw\":1.602,\"use_waypoints\":true},
    {\"task\":\"move_to\",\"x\":0.269,\"y\":-1.141,\"yaw\":3.082,\"use_waypoints\":true},
    {\"task\":\"follow_aruco\",\"marker_id\":600,\"timeout_sec\":35.0}
  ] }' }"


