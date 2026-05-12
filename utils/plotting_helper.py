# -*- coding: utf-8 -*-
"""
@Time    : 23/3/2026 10:23 am
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
from utils.env_simulator_helper import shapelypoly_to_matpoly
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import math
from shapely.geometry import LineString, Point, Polygon
import matplotlib.animation as animation
import numpy as np
from PIL import Image


def _resolve_aircraft_image_path(env):
    resources_dir = Path(__file__).resolve().parent.parent / "resources"
    candidate_paths = []

    env_resource_file = getattr(env, "resource_file", None)
    if env_resource_file:
        candidate_paths.append(Path(env_resource_file).resolve().parent)

    candidate_paths.extend([
        resources_dir,
        resources_dir / "pictures",
        resources_dir / "images",
        resources_dir / "icons",
    ])

    for base_dir in candidate_paths:
        if not base_dir.exists():
            continue
        for pattern in ("*.png", "*.PNG"):
            matches = sorted(base_dir.rglob(pattern))
            if matches:
                return matches[0]

    raise FileNotFoundError("No aircraft PNG was found under the project's resources directory.")


def _resolve_sprite_image_path(env, explicit_path=None):
    if explicit_path is not None:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate

    for attr_name in (
        "gif_sprite_texture_path",
        "occupied_poly_texture_path",
        "destination_marker_texture_path",
        "start_marker_texture_path",
    ):
        candidate = getattr(env, attr_name, None)
        if candidate is None:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    return _resolve_aircraft_image_path(env)


def _resolve_role_sprite_path(env, attr_name, explicit_path=None, fallback_attr_names=()):
    if explicit_path is not None:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate

    candidate = getattr(env, attr_name, None)
    if candidate is not None:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    for fallback_attr_name in fallback_attr_names:
        fallback_candidate = getattr(env, fallback_attr_name, None)
        if fallback_candidate is None:
            continue
        fallback_path = Path(fallback_candidate)
        if fallback_path.exists():
            return fallback_path

    return _resolve_aircraft_image_path(env)


def _crop_rgba_to_foreground(rgba_img, alpha_threshold=1e-3):
    foreground = rgba_img[..., 3] > alpha_threshold
    if not np.any(foreground):
        return rgba_img

    row_idx, col_idx = np.where(foreground)
    min_row, max_row = row_idx.min(), row_idx.max()
    min_col, max_col = col_idx.min(), col_idx.max()
    return rgba_img[min_row:max_row + 1, min_col:max_col + 1, :]


def _load_sprite_rgba(sprite_path, alpha=1.0, neutral_tolerance=0.10, light_background_threshold=0.72):
    sprite_img = Image.open(str(sprite_path)).convert("RGBA")
    rgba_img = np.asarray(sprite_img).astype(np.float32) / 255.0

    rgb = rgba_img[..., :3]
    source_alpha = rgba_img[..., 3]

    # Remove border-connected bright neutral background for JPG-like assets,
    # but preserve already-transparent PNG foreground.
    brightness = rgb.mean(axis=-1)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    candidate_background = (
        (source_alpha > 1e-3)
        & (brightness >= light_background_threshold)
        & (chroma <= neutral_tolerance)
    )

    background_mask = np.zeros(candidate_background.shape, dtype=bool)
    visited = np.zeros(candidate_background.shape, dtype=bool)
    height, width = candidate_background.shape
    stack = []

    for x in range(width):
        stack.append((0, x))
        stack.append((height - 1, x))
    for y in range(1, height - 1):
        stack.append((y, 0))
        stack.append((y, width - 1))

    while stack:
        y, x = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True
        if not candidate_background[y, x]:
            continue

        background_mask[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y < height - 1:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x < width - 1:
            stack.append((y, x + 1))

    alpha_channel = source_alpha * (~background_mask).astype(np.float32) * alpha
    rgba_result = np.dstack((rgb, alpha_channel))
    return _crop_rgba_to_foreground(rgba_result)


def _draw_sprite_clipped_to_polygon(ax, sprite_rgba, polygon, alpha=1.0, zorder=1.0, outline_color=None, outline_alpha=0.35):
    sprite_with_alpha = sprite_rgba.copy()
    sprite_with_alpha[..., 3] *= alpha

    min_x, min_y, max_x, max_y = polygon.bounds
    clip_patch = shapelypoly_to_matpoly(polygon, True, "none")
    clip_patch.set_transform(ax.transData)
    ax.imshow(
        sprite_with_alpha,
        extent=(min_x, max_x, min_y, max_y),
        interpolation="bilinear",
        clip_path=clip_patch,
        clip_on=True,
        zorder=zorder,
    )

    if outline_color is not None:
        outline_patch = shapelypoly_to_matpoly(polygon, False, outline_color)
        outline_patch.set_facecolor("none")
        outline_patch.set_linewidth(0.3)
        outline_patch.set_alpha(outline_alpha)
        outline_patch.set_zorder(zorder + 0.05)
        ax.add_patch(outline_patch)


def _draw_centered_sprite(ax, sprite_rgba, x, y, radius, alpha=1.0, zorder=3.0):
    if radius <= 0:
        return

    sprite_with_alpha = sprite_rgba.copy()
    sprite_with_alpha[..., 3] *= alpha
    height, width = sprite_with_alpha.shape[:2]
    aspect_ratio = width / float(height)
    half_height = radius
    half_width = radius * aspect_ratio

    ax.imshow(
        sprite_with_alpha,
        extent=(x - half_width, x + half_width, y - half_height, y + half_height),
        interpolation="bilinear",
        zorder=zorder,
    )


def _style_plot_axes(ax, env):
    ax.axis("equal")
    ax.set_xlim(env.bound[0], env.bound[1])
    ax.set_ylim(env.bound[2], env.bound[3])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _animate_sprite_frame(frame_num, ax, env, trajectory_eachPlay, occupied_sprite_rgba, destination_sprite_rgba, start_sprite_rgba):
    ax.clear()
    _style_plot_axes(ax, env)

    for one_poly in env.world_map_2D_polyList[0][0]:
        _draw_sprite_clipped_to_polygon(
            ax,
            occupied_sprite_rgba,
            one_poly,
            alpha=0.18,
            zorder=0.8,
            outline_color="#d8cf6a",
            outline_alpha=0.4,
        )

    for poly in env.buildingPolygons:
        matp_poly = shapelypoly_to_matpoly(poly, False, "red")
        matp_poly.set_facecolor("#c62828")
        matp_poly.set_alpha(0.6)
        matp_poly.set_linewidth(1.0)
        matp_poly.set_zorder(1.0)
        ax.add_patch(matp_poly)

    for agent_idx, agent in env.all_uavs.items():
        goal_x, goal_y = agent.goal[-1]
        _draw_centered_sprite(
            ax,
            destination_sprite_rgba,
            goal_x,
            goal_y,
            radius=env.grid_length * 0.5,
            alpha=0.9,
            zorder=2.2,
        )
        goal_label = ax.text(
            goal_x,
            goal_y + env.grid_length * 0.35,
            str(agent_idx),
            ha="center",
            va="center",
            color="white",
            fontsize=6,
            fontweight="bold",
            zorder=2.3,
        )
        goal_label.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="black")])

    for agent_idx, agent_state in enumerate(trajectory_eachPlay[frame_num]):
        x, y = agent_state[0], agent_state[1]
        _draw_centered_sprite(
            ax,
            start_sprite_rgba,
            x,
            y,
            radius=env.all_uavs[agent_idx].protectiveBound,
            alpha=0.95,
            zorder=3.0,
        )
        agent_label = ax.text(
            x,
            y + env.all_uavs[agent_idx].protectiveBound + 1.0,
            str(agent_idx),
            ha="center",
            va="center",
            color="#4dd0e1",
            fontsize=6,
            fontweight="bold",
            zorder=3.1,
        )
        agent_label.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground="black")])

    return ax.images + ax.patches + ax.texts


def animate(frame_num, ax, env, trajectory_eachPlay):
    ax.clear()
    ax.axis('equal')
    ax.set_xlim(env.bound[0], env.bound[1])
    ax.set_ylim(env.bound[2], env.bound[3])
    ax.axvline(x=env.bound[0], c="green")
    ax.axvline(x=env.bound[1], c="green")
    ax.axhline(y=env.bound[2], c="green")
    ax.axhline(y=env.bound[3], c="green")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")

    # draw occupied_poly
    for one_poly in env.world_map_2D_polyList[0][0]:
        one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y')
        ax.add_patch(one_poly_mat)
    # draw non-occupied_poly
    for zero_poly in env.world_map_2D_polyList[0][1]:
        zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
        # ax.add_patch(zero_poly_mat)

    # show building obstacles
    for poly in env.buildingPolygons:
        matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
        ax.add_patch(matp_poly)

    for agentIdx, agent in env.all_uavs.items():
        ax.plot(agent.ini_pos[0], agent.ini_pos[1],
                marker=MarkerStyle(">",
                                   fillstyle="right",
                                   transform=Affine2D().rotate_deg(math.degrees(agent.heading))),
                color='y')
        ax.text(agent.ini_pos[0], agent.ini_pos[1], agent.agent_name)
        ax.plot(agent.goal[-1][0], agent.goal[-1][1], marker='*', color='y', markersize=10)
        ax.text(agent.goal[-1][0], agent.goal[-1][1], agent.agent_name)

        # link individual drone's starting position with its goal
        ini = agent.ini_pos
        # for wp in agent.goal:
        for wp in agent.ref_line.coords:
            # plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
            ax.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
            ini = wp

    for a_idx, agent in enumerate(trajectory_eachPlay[frame_num]):
        x, y = agent[0], agent[1]
        ax.plot(x, y, 'o', color='r')

        # plt.text(x-1, y-1, 'agent_'+str(a_idx)+'_'+str(round(float(frame_num), 2)))
        ax.text(x - 1, y - 1, 'agent_' + str(a_idx) + '_' + str(agent[2]))

        self_circle = Point(x, y).buffer(env.all_uavs[0].protectiveBound, cap_style='round')
        grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
        ax.add_patch(grid_mat_Scir)

    return ax.patches + ax.texts


def save_gif(env, trajectory_eachPlay, pre_fix, episode_to_check, episode):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    fig, ax = plt.subplots(1, 1)

    ax.axis('equal')
    ax.set_xlim(env.bound[0], env.bound[1])
    ax.set_ylim(env.bound[2], env.bound[3])
    ax.axvline(x=env.bound[0], c="green")
    ax.axvline(x=env.bound[1], c="green")
    ax.axhline(y=env.bound[2], c="green")
    ax.axhline(y=env.bound[3], c="green")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")

    # draw occupied_poly
    for one_poly in env.world_map_2D_polyList[0][0]:
        one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y')
        ax.add_patch(one_poly_mat)
    # draw non-occupied_poly
    for zero_poly in env.world_map_2D_polyList[0][1]:
        zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
        # ax.add_patch(zero_poly_mat)

    # show building obstacles
    for poly in env.buildingPolygons:
        matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
        ax.add_patch(matp_poly)

    for agentIdx, agent in env.all_uavs.items():
        ax.plot(agent.ini_pos[0], agent.ini_pos[1],
                marker=MarkerStyle(">",
                                   fillstyle="right",
                                   transform=Affine2D().rotate_deg(math.degrees(agent.heading))),
                color='y')
        ax.text(agent.ini_pos[0], agent.ini_pos[1], agent.agent_name)
        # plot self_circle of the drone
        self_circle = Point(agent.ini_pos[0],
                            agent.ini_pos[1]).buffer(agent.protectiveBound, cap_style='round')
        grid_mat_Scir = shapelypoly_to_matpoly(self_circle, inFill=False)
        ax.add_patch(grid_mat_Scir)

        # plot drone's detection range
        detec_circle = Point(agent.ini_pos[0],
                             agent.ini_pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
        detec_circle_mat = shapelypoly_to_matpoly(detec_circle, inFill=False)
        ax.add_patch(detec_circle_mat)

        ax.plot(agent.goal[-1][0], agent.goal[-1][1], marker='*', color='y', markersize=10)
        ax.text(agent.goal[-1][0], agent.goal[-1][1], agent.agent_name)

    # Create animation
    ani = animation.FuncAnimation(fig, animate, fargs=(ax, env, trajectory_eachPlay), frames=len(trajectory_eachPlay),
                                  interval=300, blit=False)
    # Save as GIF
    os.makedirs(pre_fix, exist_ok=True)
    gif_path = os.path.join(
        pre_fix,
        "episode_{}_simulation_num_{}.gif".format(str(episode_to_check), int(episode)),
    )
    ani.save(gif_path, writer='pillow')

    # Close figure
    plt.close(fig)


def save_sprite_gif(
    env,
    trajectory_eachPlay,
    output_path,
    sprite_image_path=None,
    destination_sprite_image_path=None,
    start_sprite_image_path=None,
    interval_ms=300,
):
    if not trajectory_eachPlay:
        raise ValueError("trajectory_eachPlay is empty, cannot render GIF.")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    occupied_sprite_path = _resolve_role_sprite_path(
        env,
        "occupied_poly_texture_path",
        explicit_path=sprite_image_path,
        fallback_attr_names=("gif_sprite_texture_path", "destination_marker_texture_path", "start_marker_texture_path"),
    )
    destination_sprite_path = _resolve_role_sprite_path(
        env,
        "destination_marker_texture_path",
        explicit_path=destination_sprite_image_path,
        fallback_attr_names=("occupied_poly_texture_path", "gif_sprite_texture_path", "start_marker_texture_path"),
    )
    start_sprite_path = _resolve_role_sprite_path(
        env,
        "start_marker_texture_path",
        explicit_path=start_sprite_image_path,
        fallback_attr_names=("occupied_poly_texture_path", "gif_sprite_texture_path", "destination_marker_texture_path"),
    )

    occupied_sprite_rgba = _load_sprite_rgba(occupied_sprite_path, alpha=1.0)
    destination_sprite_rgba = _load_sprite_rgba(destination_sprite_path, alpha=1.0)
    start_sprite_rgba = _load_sprite_rgba(start_sprite_path, alpha=1.0)

    fig, ax = plt.subplots(1, 1)
    _style_plot_axes(ax, env)

    ani = animation.FuncAnimation(
        fig,
        _animate_sprite_frame,
        fargs=(ax, env, trajectory_eachPlay, occupied_sprite_rgba, destination_sprite_rgba, start_sprite_rgba),
        frames=len(trajectory_eachPlay),
        interval=interval_ms,
        blit=False,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ani.save(output_path, writer="pillow")
    plt.close(fig)


def view_static_traj_DWTD(env, trajectory_eachPlay, random_map_idx, save_path=None, max_time_step=None):
    if not trajectory_eachPlay:
        raise ValueError("trajectory_eachPlay is empty, cannot render static trajectory.")

    aircraft_png_path = _resolve_aircraft_image_path(env)
    plane_img = Image.open(str(aircraft_png_path)).convert("RGBA")

    w, h = plane_img.size  # Note: size returns (width, height)
    # w, h = plane_img.shape[:2]  # Note: size returns (width, height)
    aspect_ratio = w / h

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    fig, ax = plt.subplots(1, 1)
    colors = [
        (0.5, 0, 0.5),  # Purple
        (0.2, 0.8, 0.2),  # Lime
        (1, 0, 0),  # Red
        (0, 1, 0),  # Green
        (0, 0, 1),  # Blue
        (0, 1, 1),  # Cyan
        (1, 0, 1),  # Magenta
        (1, 1, 0),  # Yellow
        (1, 0.65, 0),  # Orange
    ]

    # draw occupied_poly
    world_poly_collection = getattr(env, "world_map_2D_polyList_collection", None)
    world_poly_source = world_poly_collection[random_map_idx] if world_poly_collection is not None else env.world_map_2D_polyList
    occupied_polys = world_poly_source[0][0]
    free_polys = world_poly_source[0][1]

    for one_poly in occupied_polys:
        one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y')
        ax.add_patch(one_poly_mat)
    # draw non-occupied_poly
    for zero_poly in free_polys:
        zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
        # ax.add_patch(zero_poly_mat)

    # draw trajectory in current episode
    if max_time_step is None:
        max_time_step = len(trajectory_eachPlay)
    max_time_step = min(max_time_step, len(trajectory_eachPlay))

    scale_increase = 1.5
    half_height = env.all_uavs[0].protectiveBound * scale_increase
    half_width = half_height * aspect_ratio  # scale width according to image ratio

    for agentIDX, agent in env.all_uavs.items():
        previous_position = agent.ini_pos
        for trajectory_idx in range(max_time_step):
            each_agent_traj = trajectory_eachPlay[trajectory_idx][agentIDX]
            x, y = each_agent_traj[0], each_agent_traj[1]
            if trajectory_idx == 0:
                heading = math.degrees(agent.heading)
                # plot initial point
                ax.plot(agent.ini_pos[0], agent.ini_pos[1],
                        marker=MarkerStyle(">",
                                           fillstyle="right",
                                           transform=Affine2D().rotate_deg(heading)),
                        color=colors[agentIDX % len(colors)], markersize=10, label='Origin')
                # plot  goal point
                ax.plot(agent.goal[-1][0], agent.goal[-1][1],
                        marker='*', color=colors[agentIDX % len(colors)], markersize=10,
                        label='Destination')

            if trajectory_idx >= max_time_step:
                break

            # Draw the trajectory as dotted lines starting from the initial position
            # if trajectory_idx > 0:  # Ensure we're not drawing a redundant line from ini_pos to itself
            if trajectory_idx % 2 == 0:  # Ensure we're not drawing a redundant line from ini_pos to itself
                # plt.plot([previous_position[0], x], [previous_position[1], y], linestyle=(0, (1, 10)),
                #          color=colors[agentIDX])
                # Compute alpha: 0.2 (light) to 1.0 (dark)
                if max_time_step > 1:
                    alpha = 0.2 + 0.8 * (trajectory_idx / float(max_time_step - 1))
                else:
                    alpha = 1.0

                circle = patches.Circle(
                    (x, y),
                    radius=2.5,
                    facecolor=colors[agentIDX % len(colors)],
                    edgecolor='none',
                    alpha=alpha,
                    zorder=1
                )
                ax.add_patch(circle)

            # Start with the agent's initial position
            if trajectory_idx > 0:
                prev_x, prev_y = trajectory_eachPlay[trajectory_idx - 1][agentIDX][0:2]
                heading = math.degrees(math.atan2(y - prev_y, x - prev_x))
            else:
                heading = math.degrees(agent.heading)
            # Update previous position
            previous_position = (x, y)

            if trajectory_idx == max_time_step - 1:
                # Final position with aircraft marker
                img_extent = [
                    x - half_width, x + half_width,
                    y - half_height, y + half_height
                ]
                transform = Affine2D().rotate_deg_around(x, y, heading - 90) + ax.transData
                ax.imshow(plane_img, extent=img_extent, zorder=10, transform=transform, interpolation='none')

                # Draw the protective boundary around the final position
                self_circle = Point(x, y).buffer(env.all_uavs[0].protectiveBound, cap_style='round')
                grid_mat_SCir = shapelypoly_to_matpoly(self_circle, inFill=True)
                grid_mat_SCir.set_facecolor(colors[agentIDX % len(colors)])
                grid_mat_SCir.set_edgecolor("none")
                grid_mat_SCir.set_zorder(2)
                grid_mat_SCir.set_alpha(0.9)

                ax.add_patch(grid_mat_SCir)
                ax.text(x + 3, y + 3, 'a' + str(agentIDX))

    bound_collection = getattr(env, "bound_collection", None)
    current_bound = bound_collection[random_map_idx] if bound_collection is not None else env.bound
    ax.set_xlim(current_bound[0], current_bound[1])
    ax.set_ylim(current_bound[2], current_bound[3])
    ax.set_xlabel("N-S direction (m)")
    ax.set_ylabel("E-W direction (m)")


    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        # save svg
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, bbox_inches='tight')
        # save pdf
        pdf_path = os.path.splitext(save_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        # save png
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # prevent open up all figures at end of training

    # plt.show()
