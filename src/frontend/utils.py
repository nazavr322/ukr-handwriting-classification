from collections import namedtuple
from math import ceil

import cv2 as cv


Point = namedtuple('Point', ('x', 'y'))


def create_bounding_boxes(
    metadata: list[dict], stroke_width: int = 5
) -> list[tuple[Point, Point]]:
    """
    Return list of `Point` pares. Each tuple corresponds to top-left and
    bot-right points to draw a rectangle. `stoke_width` param is a pen-width
    to take into account when computing margins.
    """
    boxes = []
    for meta in metadata:
        left, top = meta['left'], meta['top']
        width, height = meta['width'], meta['height']
        p2 = Point(
            ceil(left + width) + stroke_width,
            ceil(top + height) + stroke_width,
        )
        boxes.append((Point(int(left), int(top)), p2))
    return boxes


def merge_bounding_boxes(
    boxes: list[tuple[Point, Point]]
) -> tuple[Point, Point]:
    """
    Accepts a sequence of tuples with two `Point` objects representing the
    bounding boxes of the image.
    Return 2 `Point` objects representing top-left and bottom-right points of
    the global bounding box.
    """
    points = tuple(point for box in boxes for point in box)
    leftmost_p = min(points, key=lambda p: (p.x, p.y))
    rightmost_p = max(points, key=lambda p: (p.x, p.y))
    highest_p = min(points, key=lambda p: p.y)
    lowest_p = max(points, key=lambda p: p.y)
    p1 = Point(max(0, leftmost_p.x,), max(0, highest_p.y))
    return p1, Point(rightmost_p.x, lowest_p.y)


def encode_image(img):
    """Encodes .png image into a streaming data."""
    bgr_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return cv.imencode('.png', bgr_img)[1]
