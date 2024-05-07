"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
import pdb

box_colormap = [
    [1, 1, 1],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 1],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, pred_panoptic, kitti_cfg, category_list, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(pred_panoptic, torch.Tensor):
        pred_panoptic = pred_panoptic.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name='seg')

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    # color_dict = kitti_cfg["color_map"]
    # learning_map_inv = kitti_cfg["learning_map_inv"]
    # learning_map = kitti_cfg["learning_map"]
    # pdb.set_trace()
    # color_dict = {learning_map[key]:color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        # pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        color_dict = kitti_cfg["color_map_post"]
        point_colors = np.ones((points.shape[0], 3))
        for key, val in color_dict.items():
            mask = (pred_panoptic & 0XFFFF) == key
            point_colors[mask] = val
            # point_colors[mask] = [ele/255. for ele in val]
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)


    # pdb.set_trace()
    # ______________________________other visualization_________________________________________
    vis_ins = open3d.visualization.Visualizer()
    vis_ins.create_window(window_name='ins')
    vis_ins.get_render_option().point_size = 1.0
    vis_ins.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis_ins.add_geometry(axis_pcd)

    pts_ins = open3d.geometry.PointCloud()
    pts_ins.points = open3d.utility.Vector3dVector(points[:, :3])
    vis_ins.add_geometry(pts_ins)

    things_ins = (pred_panoptic >> 16) & 0xFFFF
    things_max = things_ins.max()
    things_min = things_ins.min()
    point_colors = np.ones((points.shape[0], 3))
    for i in range(things_min, things_max):
        # pdb.set_trace()
        ins_mask = things_ins == i
        point_colors[ins_mask] = color_dict[i%19]
    pts_ins.colors = open3d.utility.Vector3dVector(point_colors)
    
    

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 1, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (1, 0, 0), ref_labels, ref_scores)

    while True:
        vis.update_geometry(pts)
        if not vis.poll_events():
            break
        vis.update_renderer()

        vis_ins.update_geometry(pts_ins)
        if not vis_ins.poll_events():
            break
        vis_ins.update_renderer()

    vis.destroy_window()
    vis_ins.destroy_window()

    # vis.run()
    # vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i] % 7])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
