Scripts for ML for CAE 7days QuickStart Program
========================================
# Usage
```bash
# 예시: 결과 파일 result.vtu 안의 "Sxx" 스칼라 필드를 노드 기준으로 추출
python extract_field_pyvista.py result.vtu Sxx --output_csv Sxx_nodes.csv
# 예시: 변위 벡터 "U"를 추출 (U_0, U_1, U_2 + x,y,x 좌표 포함)
python extract_field_pyvista.py result.vtu U --output_csv disp_nodes.csv
```
