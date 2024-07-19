// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <Eigen/Core>
#include <memory>
#include <sophus/se3.hpp>
#include <utility>
#include <vector>

// KISS-ICP-ROS
#include "OdometryServer.hpp"
#include "Utils.hpp"

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS 2 headers
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>

// Reconfigure params
#include <memory>
#include <regex>
#include <rcl_interfaces/srv/set_parameters.hpp>
#include <rcl_interfaces/msg/parameter.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

using namespace std;

namespace {
Sophus::SE3d LookupTransform(const std::string &target_frame,
                             const std::string &source_frame,
                             const std::unique_ptr<tf2_ros::Buffer> &tf2_buffer) {
    std::string err_msg;
    if (tf2_buffer->canTransform(target_frame, source_frame, tf2::TimePointZero, &err_msg)) {
        try {
            auto tf = tf2_buffer->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
            return tf2::transformToSophus(tf);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(rclcpp::get_logger("LookupTransform"), "%s", ex.what());
        }
    }
    RCLCPP_WARN(rclcpp::get_logger("LookupTransform"), "Failed to find tf. Reason=%s",
                err_msg.c_str());
    // default construction is the identity
    return Sophus::SE3d();
}
}  // namespace

namespace kiss_icp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

OdometryServer::OdometryServer(const rclcpp::NodeOptions &options)
    : rclcpp::Node("kiss_icp_node", options) {
    base_frame_ = declare_parameter<std::string>("base_frame", base_frame_);
    odom_frame_ = declare_parameter<std::string>("odom_frame", odom_frame_);
    publish_odom_tf_ = declare_parameter<bool>("publish_odom_tf", publish_odom_tf_);
    publish_debug_clouds_ = declare_parameter<bool>("publish_debug_clouds", publish_debug_clouds_);
    position_covariance_ = declare_parameter<double>("position_covariance", 0.1);
    orientation_covariance_ = declare_parameter<double>("orientation_covariance", 0.1);

    kiss_icp::pipeline::KISSConfig config;
    config.max_range = declare_parameter<double>("max_range", config.max_range);
    config.min_range = declare_parameter<double>("min_range", config.min_range);
    config.deskew = declare_parameter<bool>("deskew", config.deskew);
    config.voxel_size = declare_parameter<double>("voxel_size", config.max_range / 100.0);
    config.max_points_per_voxel =
        declare_parameter<int>("max_points_per_voxel", config.max_points_per_voxel);
    config.initial_threshold =
        declare_parameter<double>("initial_threshold", config.initial_threshold);
    config.min_motion_th = declare_parameter<double>("min_motion_th", config.min_motion_th);
    config.max_num_iterations =
        declare_parameter<int>("max_num_iterations", config.max_num_iterations);
    config.convergence_criterion =
        declare_parameter<double>("convergence_criterion", config.convergence_criterion);
    config.max_num_threads = declare_parameter<int>("max_num_threads", config.max_num_threads);
    if (config.max_range < config.min_range) {
        RCLCPP_WARN(get_logger(),
                    "[WARNING] max_range is smaller than min_range, settng min_range to 0.0");
        config.min_range = 0.0;
    }
    
    
    /// Parameter event handler to reconfigure
    param_handler_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    
    auto cb = [this](const rcl_interfaces::msg::ParameterEvent & event) { 
        // Look for any updates to parameters in "/a_namespace" as well as any parameter changes 
        // to our own node ("this_node") 
        std::regex re("(/kiss*)"); 
        ///cout << "Entra evento" << event.node << endl;
        if (regex_match(event.node, re)) {
            // You can also use 'get_parameters_from_event' to enumerate all changes that came
            // in on this event
            auto params = rclcpp::ParameterEventHandler::get_parameters_from_event(event);
            for (auto & p : params) {
                RCLCPP_INFO(
                    this->get_logger(),
                    "cb3: Received an update to parameter \"%s\" of type: %s: \"%s\"",
                    p.get_name().c_str(),
                    p.get_type_name().c_str(),
                    p.value_to_string().c_str());

                if (strcmp(p.get_name().c_str(), "publish_debug_clouds_")==0) {
                    // publish_debug_clouds_ = p.get_value();
                }
                
                // In case sub_topic is changed we need to resubscribe and republish them
                /*if (strcmp(p.get_name().c_str(), "pub_topic")==0) { //// TODO: Esto hay que cambiarlo a los nuevos params
                    pub_topic = p.value_to_string().c_str();

                    depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(pub_topic + "/range/image", 10);
                    info_pub_ = this->create_publisher<transformer::msg::Lidar3dSensorInfo>(pub_topic + "/range/sensor_info", 10);
                    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(pub_topic + "/point_cloud", 10);///qos
                } else if (strcmp(p.get_name().c_str(), "sub_topic")==0) {
                    sub_topic = p.value_to_string().c_str();

                    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(sub_topic + "/range_image", qos, std::bind(&Transformer_Node::depth_callback, this, std::placeholders::_1));
                    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(sub_topic + "/points", qos, std::bind(&Transformer_Node::cloud_callback, this, std::placeholders::_1));
                    info_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(sub_topic + "range_info", qos, std::bind(&Transformer_Node::sensor_info_callback, this, std::placeholders::_1));
                    
                } else if (strcmp(p.get_name().c_str(), "best_effort")==0) {
                    if(p.as_bool()){
                        //qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
                    } else{
                        //qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
                    }
                }*/
            }
            // get_all_parameters();
            
        }
    };
    handle = param_handler_->add_parameter_event_callback(cb);

    flag_save_results = declare_parameter<bool>("flag_save_results", true);
    results_file_name = declare_parameter<std::string>("results_file_name", "/home/alex/ros2_ws/results/kissTestosDifo.dat");

    // Construct the main KISS-ICP odometry node
    kiss_icp_ = std::make_unique<kiss_icp::pipeline::KissICP>(config);

    // Initialize subscribers
    pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(// TODO: Cambiar el topic y a ser posible editable por parametro
        "pointcloud_topic", rclcpp::SensorDataQoS(),
        std::bind(&OdometryServer::RegisterFrame, this, std::placeholders::_1));

    // Initialize publishers
    rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("/kiss/odometry", qos);
    if (publish_debug_clouds_) {
        frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/kiss/frame", qos);
        kpoints_publisher_ =
            create_publisher<sensor_msgs::msg::PointCloud2>("/kiss/keypoints", qos);
        map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/kiss/local_map", qos);
    }

    // Initialize the transform broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf2_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf2_buffer_->setUsingDedicatedThread(true);
    tf2_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf2_buffer_);

    RCLCPP_INFO(this->get_logger(), "KISS-ICP ROS 2 odometry node initialized");

    // Open file to save results
    if (flag_save_results)///////Esto hay que ajustarlo para que se pueda modificar con el flag
    {
        if (results_file_name.size() < 5)
        {
            RCLCPP_WARN(this->get_logger(), "Invalid name for results file: ");
            RCLCPP_WARN(this->get_logger(), results_file_name.c_str());
            flag_save_results = false;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Saving results to ");
            RCLCPP_INFO(this->get_logger(), results_file_name.c_str());
            results_file.open(results_file_name);
            cout << "123456789" << endl;
        }
    }
}
// void get_all_parameters() {

//     base_frame_ = get_parameter("base_frame").get_parameter_value().get<std::string>();
//     odom_frame_ = get_parameter("odom_frame").get_parameter_value().get<std::string>();
//     publish_odom_tf_ = get_parameter("publish_odom_tf").get_parameter_value().get<bool>();
//     publish_debug_clouds_ = get_parameter("publish_debug_clouds").get_parameter_value().get<bool>();
//     position_covariance_ = get_parameter("position_covariance").get_parameter_value().get<double>();
//     orientation_covariance_ = get_parameter("orientation_covariance").get_parameter_value().get<double>();
//     config.max_range = get_parameter("max_range").get_parameter_value().get<double>();
//     config.min_range = get_parameter("min_range").get_parameter_value().get<double>();
//     config.deskew_ = get_parameter("deskew").get_parameter_value().get<bool>();
//     config.voxel_size = get_parameter("voxel_size").get_parameter_value().get<double>();
//     config.max_points_per_voxel = get_parameter("max_points_per_voxel").get_parameter_value().get<int>();
//     config.initial_threshold = get_parameter("initial_threshold").get_parameter_value().get<double>();
//     config.min_motion_th = get_parameter("min_motion_th").get_parameter_value().get<double>();
//     config.max_num_iterations = get_parameter("max_num_iterations").get_parameter_value().get<int>();
//     config.convergence_criterion = get_parameter("convergence_criterion").get_parameter_value().get<double>();
//     config.max_num_threads = get_parameter("max_num_threads").get_parameter_value().get<int>();
// }

void OdometryServer::RegisterFrame(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    const auto cloud_frame_id = msg->header.frame_id;
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = GetTimestamps(msg);

    // Register frame, main entry point to KISS-ICP pipeline
    const auto &[frame, keypoints] = kiss_icp_->RegisterFrame(points, timestamps);

    // Extract the last KISS-ICP pose, ego-centric to the LiDAR
    const Sophus::SE3d kiss_pose = kiss_icp_->pose();

    // Spit the current estimated pose to ROS msgs handling the desired target frame
    PublishOdometry(kiss_pose, msg->header);
    // Publishing these clouds is a bit costly, so do it only if we are debugging
    if (publish_debug_clouds_) {
        PublishClouds(frame, keypoints, msg->header);
    }
}

void OdometryServer::PublishOdometry(const Sophus::SE3d &kiss_pose,
                                     const std_msgs::msg::Header &header) {
    // If necessary, transform the ego-centric pose to the specified base_link/base_footprint frame
    const auto cloud_frame_id = header.frame_id;
    const auto egocentric_estimation = (base_frame_.empty() || base_frame_ == cloud_frame_id);
    const auto pose = [&]() -> Sophus::SE3d {
        if (egocentric_estimation) return kiss_pose;
        const Sophus::SE3d cloud2base = LookupTransform(base_frame_, cloud_frame_id, tf2_buffer_);
        return cloud2base * kiss_pose * cloud2base.inverse();
    }();

    // Broadcast the tf ---
    if (publish_odom_tf_) {
        geometry_msgs::msg::TransformStamped transform_msg;
        transform_msg.header.stamp = header.stamp;
        transform_msg.header.frame_id = odom_frame_;
        transform_msg.child_frame_id = egocentric_estimation ? cloud_frame_id : base_frame_;
        transform_msg.transform = tf2::sophusToTransform(pose);
        tf_broadcaster_->sendTransform(transform_msg);
    }

    // publish odometry msg
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = header.stamp;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = cloud_frame_id;
    odom_msg.pose.pose = tf2::sophusToPose(pose);
    odom_msg.pose.covariance.fill(0.0);
    odom_msg.pose.covariance[0] = position_covariance_;
    odom_msg.pose.covariance[7] = position_covariance_;
    odom_msg.pose.covariance[14] = position_covariance_;
    odom_msg.pose.covariance[21] = orientation_covariance_;
    odom_msg.pose.covariance[28] = orientation_covariance_;
    odom_msg.pose.covariance[35] = orientation_covariance_;
    odom_publisher_->publish(std::move(odom_msg));

    // Save results to file
		if (flag_save_results)
		{
			rclcpp::Time timestamp(header.stamp);
			char timestr[20];
			snprintf(timestr, sizeof(timestr), "%.9f", timestamp.seconds());
			results_file << timestr << " "
			 			 << odom_msg.pose.pose.position.x << " " << odom_msg.pose.pose.position.y << " " << odom_msg.pose.pose.position.z << " "
			 			 << odom_msg.pose.pose.orientation.x << " " << odom_msg.pose.pose.orientation.y << " " << odom_msg.pose.pose.orientation.z << " " << odom_msg.pose.pose.orientation.w
						 << std::endl;
		}
}

void OdometryServer::PublishClouds(const std::vector<Eigen::Vector3d> frame,
                                   const std::vector<Eigen::Vector3d> keypoints,
                                   const std_msgs::msg::Header &header) {
    const auto kiss_map = kiss_icp_->LocalMap();
    const auto kiss_pose = kiss_icp_->pose().inverse();

    frame_publisher_->publish(std::move(EigenToPointCloud2(frame, header)));
    kpoints_publisher_->publish(std::move(EigenToPointCloud2(keypoints, header)));
    map_publisher_->publish(std::move(EigenToPointCloud2(kiss_map, kiss_pose, header)));
}
}  // namespace kiss_icp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(kiss_icp_ros::OdometryServer)
