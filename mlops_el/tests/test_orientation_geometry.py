import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_orientation_variant import rotate180_box


def test_rotation_preserves_size_and_is_its_own_inverse():
    box=[3630,735,402,501]
    rotated=rotate180_box(box,4032,3024)
    assert rotated==[0,1788,402,501]
    assert rotate180_box(rotated,4032,3024)==box


def test_full_image_and_edges_remain_in_bounds():
    assert rotate180_box([0,0,720,720],720,720)==[0,0,720,720]
    assert rotate180_box([0,0,20,10],100,80)==[80,70,20,10]
