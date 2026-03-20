# -*- coding: utf-8 -*-
"""
@Time    : 19/3/2026 11:11 am
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import math
import numpy as np
from matplotlib.patches import Polygon as matPolygon
from shapely.geometry import LineString, Point, Polygon
import statistics
from shapely.ops import nearest_points


def total_length_to_end_of_line(initial_point, linestring):
    """
    Calculate the total distance from an initial point to the nearest point on the line and
    from there to the end of the line.

    Parameters:
    initial_point (tuple): The initial point as (x, y).
    linestring (LineString): The LineString object.

    Returns:
    float: The total distance from the initial point to the end of the LineString.
    """
    # Create a Point object from the tuple
    point = Point(initial_point)

    # Find the nearest point on the line to the initial point
    nearest_point_on_line = linestring.interpolate(linestring.project(point))

    # Calculate the distance from the initial point to the nearest point on the line
    distance_to_line = point.distance(nearest_point_on_line)

    # Calculate the distance from the nearest point on the line to the end of the line
    projected_distance = linestring.project(nearest_point_on_line)
    distance_to_end_of_line = linestring.length - projected_distance

    # Sum the distances to get the total distance
    total_distance = distance_to_line + distance_to_end_of_line

    return total_distance


def calculate_bearing(x_host, y_host, x_intruder, y_intruder):
    delta_x = x_intruder - x_host
    delta_y = y_intruder - y_host

    theta_radians = math.atan2(delta_y, delta_x)
    theta_degrees = math.degrees(theta_radians)

    # Convert to bearing as specified
    if theta_degrees < 0:
        bearing = -theta_degrees
    else:
        bearing = 360 - theta_degrees

    return bearing


def coordinate_to_meter(target, max_, min_, span):
    portion = max_ - min_
    meter_per_unit = portion / span  # this is the length in meter represented by each unit of input
    meter = (target - min_) * meter_per_unit
    return meter  # conversion of the targeted coordinate into meter


def compute_t_cpa_d_cpa_potential_col(other_pos, host_pos, other_vel, host_vel, other_bound, host_bound, total_possible_conf):
    rel_dist_withNeg = -1 * (other_pos - host_pos)  # relative distance, host-intru
    rel_vel = other_vel - host_vel  # get relative velocity, host-intru
    rel_vel_norm_withSQ = np.square(np.linalg.norm(rel_vel))  # square of norm
    if rel_vel_norm_withSQ == 0:  # meaning this neigh with host drone relative vel = 0, same spd
        tcpa = -10
        # check possible collision manually
        time_to_check = 1  # Check for collision after t seconds
        new_nei_pos = other_pos + (other_vel * time_to_check)
        new_host_pos = host_pos + (host_vel * time_to_check)
        d_tcpa = np.linalg.norm(new_host_pos - new_nei_pos)
        if d_tcpa < (other_bound + host_bound):
            total_possible_conf = total_possible_conf + 1

    else:
        tcpa = np.dot(rel_dist_withNeg, rel_vel) / rel_vel_norm_withSQ
        d_tcpa = np.linalg.norm(((rel_dist_withNeg * -1) + (rel_vel * tcpa)))

    if (tcpa <= 1) and (tcpa >= 0) and (
            d_tcpa < (other_bound + host_bound)):
        total_possible_conf = total_possible_conf + 1
    return (tcpa, d_tcpa, total_possible_conf)


def cross_track_error(point, line):
    # Find the nearest point on the line to the given point
    nearest_pt = nearest_points(point, line)[1]

    # Calculate the cross-track distance
    distance = point.distance(nearest_pt)

    # Calculate the x and y components of the cross-track error
    x_error = abs(point.x - nearest_pt.x)
    y_error = abs(point.y - nearest_pt.y)

    return distance, x_error, y_error, nearest_pt


def initialize_3d_array_environment(girdLength, maxX, maxY, maxZ):  # grid is a cube
    arrlength_x = math.ceil(maxX / girdLength)
    arrlength_y = math.ceil(maxY / girdLength)
    arrlength_z = math.ceil(maxZ / girdLength)
    initialized3DArray = np.zeros((arrlength_x, arrlength_y, arrlength_z))
    return initialized3DArray


def shapelypoly_to_matpoly(shapelyPolgon, inFill=False, inEdgecolor='black'):
    xcoo, ycoo = shapelyPolgon.exterior.coords.xy
    matPolyConverted = matPolygon(xy=list(zip(xcoo, ycoo)), fill=inFill, edgecolor=inEdgecolor)
    return matPolyConverted


def square_grid_intersection(strTreePolyset, gridToTest, buildingPolygonDict):
    occupied = 0
    height = 0
    polygons_in_vicinity_index = strTreePolyset.query(gridToTest)  # will return possible polygons around the tested grids, including the tested grid itself. Be careful of double counting!
    if len(polygons_in_vicinity_index) == 0:
        return occupied, height

    if len(polygons_in_vicinity_index) == 1:
        possiblePoly = strTreePolyset.geometries.take(polygons_in_vicinity_index).tolist()  # this is shapely polygon
        matp_PolyConvert = shapelypoly_to_matpoly(possiblePoly[0])
        matp_gridToTest = shapelypoly_to_matpoly(gridToTest, True)


        # matplotlib.use('Qt5Agg')
        # fig, ax = plt.subplots(1, 1)
        # # Add the polygon to the axis
        # ax.add_patch(matPolyConvert)
        # ax.add_patch(gridToTest)
        # plt.autoscale()
        # # Display the plot
        # plt.show()

        if possiblePoly[0].disjoint(gridToTest):  # one possible polygon around, but does not intersect or equal
            pass
        else:
            occupied = 1
            height = buildingPolygonDict[id(possiblePoly[0])]
    else:  # if current gridToTest have spatial relationship wih two or more building polygons
        heightToAverage = []
        for possiblePoly_idx in polygons_in_vicinity_index:
            possiblePoly = strTreePolyset.geometries.take(possiblePoly_idx)  # this is shapely polygon
            if possiblePoly.disjoint(gridToTest):
                pass  # disjoint, no action required
            else:
                occupied = 1
                heightToAverage.append(buildingPolygonDict[id(possiblePoly)])
        # after look through two possible polygons and no spatial relationship between the gridToTest we can just return the result
        if occupied == 0:
            return occupied, height
        if len(heightToAverage) > 0:
            height = statistics.mean(heightToAverage)
        else:  # "heightToAverage" only has a single item
            height = heightToAverage[0]
    return occupied, height


class OUNoise:

    def __init__(self, action_dimension, largest_Nsigma=0.5, smallest_Nsigma=0.15, ini_sigma=0.15, mu=0, theta=0.15):  # sigma is the initial magnitude of the OU_noise
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = ini_sigma
        self.largest_sigma = largest_Nsigma
        self.smallest_sigma = smallest_Nsigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class NormalizeData:
    def __init__(self, x_min_max, y_min_max, spd_max, acc_range):
        self.normalize_max = 1
        self.normalize_min = -1
        self.dis_min_x = x_min_max[0]
        self.dis_max_x = x_min_max[1]
        self.dis_min_y = y_min_max[0]
        self.dis_max_y = y_min_max[1]
        self.spd_max = spd_max
        self.acc_min = acc_range[0]
        self.acc_max = acc_range[1]
        self.scale_attribute()

    def scale_attribute(self):
        self.x_scale = (self.normalize_max - self.normalize_min) / (self.dis_max_x - self.dis_min_x)
        self.y_scale = (self.normalize_max - self.normalize_min) / (self.dis_max_y - self.dis_min_y)

    def nmlz_pos(self, pos_c):
        x, y = pos_c[0], pos_c[1]
        x_normalized = 2 * ((x - self.dis_min_x) / (self.dis_max_x - self.dis_min_x)) - 1
        y_normalized = 2 * ((y - self.dis_min_y) / (self.dis_max_y - self.dis_min_y)) - 1
        return np.array([x_normalized, y_normalized])

    def reverse_nmlz_pos(self, norm_pos_c):
        norm_x, norm_y = norm_pos_c[0], norm_pos_c[1]
        x = ((norm_x + 1) / 2) * (self.dis_max_x - self.dis_min_x) + self.dis_min_x
        y = ((norm_y + 1) / 2) * (self.dis_max_y - self.dis_min_y) + self.dis_min_y
        return np.array([x, y])

    def scale_pos(self, pos_c):
        x_normalized = self.normalize_min + (pos_c[0] - self.dis_min_x) * self.x_scale
        y_normalized = self.normalize_min + (pos_c[1] - self.dis_min_y) * self.y_scale
        return np.array([x_normalized, y_normalized])

    def norm_scale(self, change_in_pos):
        return np.array([self.x_scale * change_in_pos[0], self.y_scale * change_in_pos[1]])

    def nmlz_pos_diff(self, diff):
        dx, dy = diff[0], diff[1]
        dx_min = self.dis_min_x - self.dis_max_x
        dx_max = self.dis_max_x - self.dis_min_x
        dy_min = self.dis_min_y - self.dis_max_y
        dy_max = self.dis_max_y - self.dis_min_y
        dx_normalized = 2 * ((dx - dx_min) / (dx_max - dx_min)) - 1
        dy_normalized = 2 * ((dy - dy_min) / (dy_max - dy_min)) - 1
        return dx_normalized, dy_normalized

    def nmlz_vel(self, cur_vel):
        vx, vy = cur_vel[0], cur_vel[1]
        vx_normalized = vx / self.spd_max
        vy_normalized = vy / self.spd_max
        return np.array([vx_normalized, vy_normalized])

    def reverse_nmlz_vel(self, norm_vel):
        norm_vx, norm_vy = norm_vel[0], norm_vel[1]
        vx = norm_vx * self.spd_max
        vy = norm_vy * self.spd_max
        return np.array([vx, vy])

    def nmlz_acc(self, cur_acc):
        ax, ay = cur_acc[0], cur_acc[1]
        ax_normalized = ax / self.acc_max
        ay_normalized = ay / self.acc_max
        return np.array([ax_normalized, ay_normalized])
