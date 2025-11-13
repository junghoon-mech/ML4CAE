#!/usr/bin/env python3
# extract_field_pyvista.py

import pyvista as pv
import numpy as np
import pandas as pd
import argparse

def extract_field(
    input_file: str,
    field_name: str,
    output_csv: str = "field_data.csv",
    include_coords: bool = True,
):
    """
    input_file: CAE result file (.vtu, .vtk, .exo, ...)
    field_name: point_data 또는 cell_data에 존재하는 필드 이름
    output_csv: 저장할 CSV 경로
    include_coords: 노드 좌표를 함께 저장할지 여부
    """
    mesh = pv.read(input_file)
    print(mesh)

    # 우선 point_data에서 찾고, 없으면 cell_data 검색
    if field_name in mesh.point_data:
        data = mesh.point_data[field_name]
        data_type = "point_data"
    elif field_name in mesh.cell_data:
        data = mesh.cell_data[field_name]
        data_type = "cell_data"
    else:
        raise KeyError(f"Field '{field_name}' not found in point_data or cell_data.")

    print(f"Found field '{field_name}' in {data_type} with shape {np.array(data).shape}")

    # data가 vector인지 scalar인지에 따라 처리
    data = np.asarray(data)
    n = data.shape[0]

    if data.ndim == 1:
        # scalar
        df = pd.DataFrame({field_name: data})
    else:
        # vector or tensor → 각 성분을 열로 분리
        n_comp = data.shape[1]
        cols = [f"{field_name}_{i}" for i in range(n_comp)]
        df = pd.DataFrame(data, columns(cols))

    # point_data일 때 좌표 포함(옵션)
    if include_coords and data_type == "point_data":
        coords = np.asarray(mesh.points)  # (N, 3)
        df["x"] = coords[:, 0]
        df["y"] = coords[:, 1]
        df["z"] = coords[:, 2]

    df.to_csv(output_csv, index=False)
    print(f"Saved field data to {output_csv}")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract field from CAE result using pyvista")
    parser.add_argument("input_file", type=str, help="Path to .vtu/.vtk/.exo etc.")
    parser.add_argument("field_name", type=str, help="Field name to extract")
    parser.add_argument("--output_csv", type=str, default="field_data.csv", help="Output CSV path")
    parser.add_argument("--no_coords", action="store_true", help="Do not include coordinates even for point_data")
    args = parser.parse_args()

    extract_field(
        input_file=args.input_file,
        field_name=args.field_name,
        output_csv=args.output_csv,
        include_coords=not args.no_coords,
    )
