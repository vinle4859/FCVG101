"""Generate 16 SYNTHETIC camera views purely for classroom testing.
Not a replacement for checkerboard photos from a real camera.
"""
from pathlib import Path
import cv2
import numpy as np


def main():
    rng = np.random.default_rng(42)
    W, H = 960, 720
    cols, rows = 10, 7  # squares -> 9 by 6 INTERNAL CORNERS
    square_m = .018
    sq_px = 70
    texture = np.full((rows*sq_px, cols*sq_px, 3), 250, np.uint8)
    for r in range(rows):
        for c in range(cols):
            color = 30 if ((r+c) % 2) else 235
            cv2.rectangle(texture,(c*sq_px,r*sq_px),((c+1)*sq_px-1,(r+1)*sq_px-1),
                          (color,)*3,-1)
    K = np.array([[900.,0,480.],[0,890.,360.],[0,0,1.]])
    dist = np.array([-0.14,0.04,0.001,-0.0005,-0.005],dtype=np.float64)
    out = Path('data/calibration'); out.mkdir(parents=True, exist_ok=True)
    (out / 'README_SYNTHETIC.txt').write_text('SYNTHETIC DATA, NOT REAL CAMERA EVIDENCE.\n', encoding='utf-8')
    obj_outer = np.float32([[0,0,0],[cols*square_m,0,0],
                             [cols*square_m,rows*square_m,0],
                             [0,rows*square_m,0]])
    src = np.float32([[0,0],[cols*sq_px-1,0],
                      [cols*sq_px-1,rows*sq_px-1],[0,rows*sq_px-1]])
    yy, xx = np.indices((H,W), np.float32)
    distorted_pixels = np.stack([xx,yy],axis=-1).reshape(-1,1,2)
    ideal_pixels = cv2.undistortPoints(distorted_pixels,K,dist,P=K).reshape(H,W,2)

    for i in range(16):
        # Tilt and distance variation with coverage of different image locations.
        rvec = rng.uniform(-.32,.32,3).astype(np.float64)
        tvec = np.array([-.09+rng.uniform(-.05,.05),
                         -.063+rng.uniform(-.04,.04),
                         rng.uniform(.42,.76)])
        uv, _ = cv2.projectPoints(obj_outer,rvec,tvec,K,np.zeros(5))
        dst = uv.reshape(4,2).astype(np.float32)
        Hmat = cv2.getPerspectiveTransform(src,dst)
        ideal = cv2.warpPerspective(texture,Hmat,(W,H),flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,borderValue=(185,185,185))
        bent = cv2.remap(ideal, ideal_pixels[...,0], ideal_pixels[...,1],
                         interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(185,185,185))
        if i % 4 == 0:
            bent = cv2.GaussianBlur(bent,(3,3),.45)
        noise = rng.normal(0,0.8,bent.shape)
        bent = np.uint8(np.clip(bent.astype(np.float32)+noise,0,255))
        cv2.imwrite(str(out/f'synthetic_view_{i+1:02d}.jpg'),bent,[cv2.IMWRITE_JPEG_QUALITY,95])
    print('Created 16 synthetic checkerboard views:',out.resolve())
    print('SYNTHETIC dataset only: never present values as real camera calibration.')

if __name__=='__main__': main()
