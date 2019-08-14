import numpy as np


def get_K_inv_dot_xy_1(h=192, w=256):
    focal_length = 517.97
    offset_x = 320
    offset_y = 240

    K = [[focal_length, 0, offset_x],
         [0, focal_length, offset_y],
         [0, 0, 1]]

    K_inv = np.linalg.inv(np.array(K))

    K_inv_dot_xy_1 = np.zeros((3, h, w))

    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640
                
            ray = np.dot(K_inv,
                         np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1[:, y, x] = ray[:, 0]

    return K_inv_dot_xy_1


K_inv_dot_xy_1 = get_K_inv_dot_xy_1()


# https://github.com/art-programmer/PlaneNet/blob/88e8c8d7e527ce61620b700babc8232de8804f55/code/utils.py#L860
def writePLYFileDepth(folder, index, depth, segmentation):
    h, w = 192, 256
    imageFilename = str(index) + '_segmentation_pred_blended_0.png'

    # create face from segmentation
    faces = []
    for y in range(h-1):
        for x in range(w-1):
            segmentIndex = segmentation[y, x]
            # ignore non planar region
            if segmentIndex == 0:
                continue

            # add face if three pixel has same segmentatioin
            depths = [depth[y][x], depth[y + 1][x], depth[y + 1][x + 1]]
            if segmentation[y + 1, x] == segmentIndex and segmentation[y + 1, x + 1] == segmentIndex and min(depths) > 0 and max(depths) < 10:
                faces.append((x, y, x, y + 1, x + 1, y + 1))

            depths = [depth[y][x], depth[y][x + 1], depth[y + 1][x + 1]]
            if segmentation[y][x + 1] == segmentIndex and segmentation[y + 1][x + 1] == segmentIndex and min(depths) > 0 and max(depths) < 10:
                faces.append((x, y, x + 1, y + 1, x + 1, y))

    with open(folder + '/' + str(index) + '_model.ply', 'w') as f:
        header = """ply
format ascii 1.0
comment VCGLIB generated
comment TextureFile """
        header += imageFilename
        header += """
element vertex """
        header += str(h * w)
        header += """
property float x
property float y
property float z
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_indices
property list uchar float texcoord
end_header
"""
        f.write(header)
        for y in range(h):
            for x in range(w):
                segmentIndex = segmentation[y][x]
                if segmentIndex == 20:
                    f.write("0.0 0.0 0.0\n")
                    continue
                ray = K_inv_dot_xy_1[:, y, x]
                X, Y, Z = ray * depth[y, x]
                f.write(str(X) + ' ' + str(Y) + ' ' + str(Z) + '\n')

        for face in faces:
            f.write('3 ')
            for c in range(3):
                f.write(str(face[c * 2 + 1] * w + face[c * 2]) + ' ')
            f.write('6 ')
            for c in range(3):
                f.write(str(float(face[c * 2]) / w) + ' ' + str(1 - float(face[c * 2 + 1]) / h) + ' ')
            f.write('\n')
        f.close()
    return
