import cv2
import numpy as np
import torch
from ..transforms import force_hw3, tonumpy

class OpenCVRenderer:
    """A frame by frame video renderer using OpenCV.

    Example:
    ```
    # create a renderer
    with OpenCVRenderer('out.mp4', fps = 60) as renderer:

        # add 1000 640x320 frames with random values
        for i in range(1000):
            frame = np.random.uniform(0, 255, size = (640, 320, 3))
            renderer.add_frame(frame)

    # if you haven't used a context, you have to release the renderer, which makes the video file openable
    renderer.release()
    ```
    """

    def __init__(
        self,
        outfile,
        fps = 60,
        codec="mp4v",
    ):
        """Create a video renderer.

        :param outfile: Path to the file to write the video to. The file will be created when first frame is added.
        :param fps: Frames per second.
        :param codec: Video codec, default is 'mp4v'
        """
        if fps < 1: raise ValueError(f"FPS must be at least 1, got {fps}")
        if not outfile.lower().endswith(".mp4"):
            outfile += ".mp4"

        self.outfile = outfile
        self.fps = fps
        self.codec = codec

        self.writer = None

    def add_frame(self, frame: np.ndarray | torch.Tensor):
        """Write the next frame to the video file.
        All frames must of the same shape, have np.uint8 or torch.uint8 data type.

        :param frame: Frame in np.uint8 data type.
        :raises ValueError: If frame shape is different from the previous one.
        """
        # make sure it is hw3
        frame = tonumpy(force_hw3(frame))[:,:,::-1]

        # on first frame create writer and use frame shape as video size
        if self.writer is None:
            self.shape = frame.shape

            self.writer = cv2.VideoWriter(
                self.outfile, cv2.VideoWriter_fourcc(*self.codec), self.fps, (self.shape[1], self.shape[0])
            )

        # check frame shape (opencv doesn't have that check, it just skips the frame)
        if frame.ndim != 3:
            raise ValueError(
                f"Frame must have 3 dimensions: (H, W, 3), got frame of shape {frame.shape}"
            )
        if frame.shape[2] != 3:
            raise ValueError(
                f"The last frame dimension must be 3 (RGB), got frame of shape {frame.shape}"
            )
        if frame.shape != self.shape:
            raise ValueError(
                f"Frame size {frame.shape} is different from previous frame size {self.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame must be of type np.uint8, got {frame.dtype}")

        # write new frame to file
        self.writer.write(frame)

    def release(self):
        """Close the writer, releasing access to the video file."""
        if self.writer is None:
            raise ValueError("No frames have been added to this renderer.")
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()


def render_frames(file, frames, fps = 60):
    with OpenCVRenderer(file, fps) as r:
        for frame in frames:
            r.add_frame(frame)

