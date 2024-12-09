from pathlib import Path
import argparse
import cv2
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="./",
    )
    parser.add_argument(
        "--vid_path",
        default="",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=2,
    )


    args = parser.parse_args()

    vid_path = Path(args.vid_path)
    out_dir = Path(args.outdir) / "images"

    out_dir.mkdir(exist_ok=True, parents=True)

    n = 0
    file_id = 0

    cap = cv2.VideoCapture(str(vid_path))
    success = True
    while success:
        success, img = cap.read()

        n += 1
        if success and n % args.subsample == 0:
            img = cv2.resize(img, (0,0), fx=1./args.downscale, fy=1./args.downscale)
            cv2.imwrite(str(out_dir / f"{file_id:04d}.jpg"), img)
            file_id += 1


    





