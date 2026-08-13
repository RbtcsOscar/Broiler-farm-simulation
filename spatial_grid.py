"""
spatial_grid.py
===============
1,500 마리의 모든 pair 를 직접 계산하지 않기 위한 uniform spatial grid (가이드 §24).

cell_size 는 닭-닭 상호작용 최대 반경(R_SOCIAL, R_C, 2 r_c) 이상으로 잡으므로,
각 닭은 자신의 cell 과 주변 8개 cell(=3x3) 만 검색하면 필요한 이웃을 모두 얻는다.

사용처 (공통):
    1) 닭-닭 collision
    2) social clustering (nearest neighbor)
    3) ESCAPE propagation C_i
"""

import numpy as np

# 3x3 이웃 offset (자기 cell 포함)
_OFFSETS = [(-1, -1), (0, -1), (1, -1),
            (-1,  0), (0,  0), (1,  0),
            (-1,  1), (0,  1), (1,  1)]


class SpatialGrid:
    def __init__(self, width, height, cell_size):
        self.cell = cell_size
        self.nx = int(np.ceil(width / cell_size)) + 1
        self.ny = int(np.ceil(height / cell_size)) + 1
        self.cell_dict = {}          # (cx, cy) -> np.ndarray[int] of chicken indices

    def build(self, x, y):
        """현재 위치로 grid 를 재구성한다 (§26-1)."""
        cx = np.clip((x / self.cell).astype(np.int32), 0, self.nx - 1)
        cy = np.clip((y / self.cell).astype(np.int32), 0, self.ny - 1)

        # cell key 로 정렬 → 연속 구간으로 분할하면 O(N log N) 로 dict 구성
        key = cx.astype(np.int64) * self.ny + cy
        order = np.argsort(key, kind="stable")
        key_sorted = key[order]
        boundaries = np.nonzero(np.diff(key_sorted))[0] + 1
        groups = np.split(order, boundaries) if order.size else []

        cell_dict = {}
        for g in groups:
            i0 = g[0]
            cell_dict[(int(cx[i0]), int(cy[i0]))] = g
        self.cell_dict = cell_dict

    def neighborhoods(self):
        """occupied cell 마다 (members, neighborhood_members) 를 yield.

        members              : 그 cell 의 닭 인덱스 (각 닭은 정확히 1회 member)
        neighborhood_members : 3x3 cell 전체의 닭 인덱스 (자기 자신 포함)
        """
        cd = self.cell_dict
        for (cxi, cyi), members in cd.items():
            neigh_lists = []
            for dx, dy in _OFFSETS:
                arr = cd.get((cxi + dx, cyi + dy))
                if arr is not None:
                    neigh_lists.append(arr)
            neigh = np.concatenate(neigh_lists) if neigh_lists else members
            yield members, neigh
