import cv2
import numpy as np
from scipy.spatial.distance import euclidean as distance
import sys

def optical_flow(img0, img1):
    flow = None
    pyr_scale = 0.5
    levels = 4
    winsize = 20
    iterations = 5
    poly_n = 5
    poly_sigma = 1.2
    flags = 0 #cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    flow = cv2.calcOpticalFlowFarneback(img0, img1, flow, pyr_scale, levels,
                            winsize, iterations, poly_n, poly_sigma, flags)
    return flow

def optical_flow_from_file(file):
    flow = None
    with open(file, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print("Invalid .flo file")
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2*w*h)
            flow = np.resize(data, (h, w, 2))
    return flow

def inpaint(img, mask):
    inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    return inpaint

def bilinear(position, grid):
    shape = np.shape(grid);
    range_x = range(shape[0])
    range_y = range(shape[1])
    x = position[0]
    y = position[1]
    xi = int(x)
    yi = int(y)
    tx = x - xi
    ty = y - yi
    if xi in range_x and yi in range_y:
        c00 = grid[xi, yi]
        c01 = grid[xi, yi + 1] if yi + 1 in range_y else c00
        c10 = grid[xi + 1, yi] if xi + 1 in range_x else c00
        c11 = grid[xi + 1, yi + 1] if xi + 1 in range_x and yi + 1 in range_y else c00
        a = c00 * (1 - tx) + c10 * tx
        b = c01 * (1 - tx) + c11 * tx
        result = a * (1 - ty) + b * ty
    else:
        result = np.zeros([shape[2]])

    return result

def displayable_flow(flow):
    shape = [np.shape(flow)[0], np.shape(flow)[1], 3]
    hsv = np.zeros(shape, dtype=np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180 / np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    displayable_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    return displayable_flow

def get_splat_uv(u, v, urange, vrange, flow):
    offsets = [-0.5, 0.5]
    pos_u = u + flow[0]
    pos_v = v + flow[1]
    splats = []
    for offset_u in offsets:
        for offset_v in offsets:
            splat_u = int(pos_u + offset_u)
            splat_v = int(pos_v + offset_v)
            splat_uv = [splat_u, splat_v]
            if splat_u in urange and splat_v in vrange and splat_uv not in splats:
                splats.append(splat_uv)
    
    return splats

def interpolate(img0, img1, t):
    shape = np.shape(img0)
    h = shape[0]
    w = shape[1]

    imgs = [img0, img1]

    img_interp = np.zeros([2, h, w, 3], dtype=np.uint8)
    img_interp[...] = [255, 0, 255]

    img0_gs = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1_gs = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    flows = np.zeros([2, h, w, 2], dtype=np.float32)
    flows[0] = optical_flow(img0_gs, img1_gs) # forward optical flow
    flows[1] = optical_flow(img1_gs, img0_gs) # backward optical flow
    cv2.imwrite("flowforward.png", displayable_flow(flows[0]))
    cv2.imwrite("flowbackward.png", displayable_flow(flows[1]))

    color_distance_min = np.ones([2, h, w], dtype=np.float32) * np.finfo(np.float32).max
    inpaint_masks = np.ones([2, h, w], dtype=np.uint8)

    urange = range(h)
    vrange = range(w)
    for direction in [0, 1]:
        for ui in urange:
            for vi in vrange:
                i = direction
                j = (direction + 1) % 2

                alpha = np.empty(2);
                alpha[i] = abs(i - t)
                alpha[j] = abs(j - t)
                flow = flows[i, ui, vi]
                
                uj = int(ui + flow[0] * t)
                vj = int(vi + flow[1] * t)
                
                c = np.empty([2, 3])
                c[i] = c[j] = imgs[i][ui, vi]
                if uj in urange and vj in vrange:
                    c[j] = imgs[j][uj, vj] #bilinear([ui + flow[0], vi + flow[1]], imgs[j])

                color_distance = distance(c[i], c[j])
                splats = get_splat_uv(ui, vi, urange, vrange, t * flow)
                for splat_uv in splats:
                    u = splat_uv[0]
                    v = splat_uv[1]
                    if color_distance < color_distance_min[direction, u, v]:
                        color_distance_min[direction, u, v] = color_distance
                        inpaint_masks[direction, u, v] = 0
                        d = np.empty(2);
                        for index in [0, 1]:
                            index_other = (index + 1) % 2
                            a = - alpha[index] * flow;
                            b = alpha[index] * bilinear([u, v] + a, flows[index])
                            d[index] = np.linalg.norm(a + b)
                        
                        min_index = i if d[i] < d[j] else j
                        max_index = (min_index + 1) % 2      
                        threshold = 2;
                        if d[max_index] > 0.5:
                            ct = c[min_index]
                        else:
                            ct = alpha[i] * c[i] + alpha[j] * c[j]
                        ct = alpha[i] * c[i] + alpha[j] * c[j]
                        img_interp[direction, u, v] = ct
        img_interp[direction] = inpaint(img_interp[direction], inpaint_masks[direction])
    result = np.zeros([h, w, 3], dtype=np.uint8)
    for u in urange:
        for v in vrange:
            result[u, v] = 0.5 * img_interp[0, u, v] + 0.5 * img_interp[1, u, v]

    """
    cv2.imwrite("frame_1pred_0.png", img_interp[0])
    cv2.imwrite("frame_1pred_1.png", img_interp[1])
    cv2.imwrite("flowforward.png", displayable_flow(flows[0]))
    cv2.imwrite("flowbackward.png", displayable_flow(flows[1]))
    cv2.imwrite("frame_1pred.png", result)
    """
    return result

def main():
    flow_from_file = optical_flow_from_file("flow10.flo")
    cv2.imwrite("flow_from_file.png", displayable_flow(flow_from_file))

    frame0 = cv2.imread("frame_1.png", cv2.IMREAD_COLOR)
    frame1 = cv2.imread("frame_2.png", cv2.IMREAD_COLOR)
    interp_img = interpolate(frame0, frame1, 0.5)
    """
    stream = cv2.VideoCapture(sys.argv[1])
    ret, frame0 = stream.read()
    prev = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    frame_num = 1
    while True:
        ret, frame1 = stream.read()
        if not ret:
            stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        interp_img = interpolate(frame0, frame1, 0.5)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        frame0 = frame1
        frame_num = frame_num + 1
    stream.release()
    cv2.destroyAllWindows()
    """

if __name__ == '__main__':
    main()