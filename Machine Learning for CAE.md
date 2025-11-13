# CAE 기반 Machine Learning — 7일 체크리스트

## Day 1 — CAE 데이터는 왜 ML로 어려운가
- [ ] CAE 데이터 구조 복습(geometry / mesh / field / BC)
- [ ] CAE result nodal/element 구조 이해
- [ ] High-dimensional data 특성 이해
- [ ] Irregular mesh의 ML 적용 어려움 설명 가능
- [ ] CAE → ML 데이터 유형 4가지 정리
- [ ] ML 적용 목적 정리(서로게이트, geometry → property 등)
### 실습
- [ ] Python으로 CAE result 1개 로드하고 field/nodal 데이터 구조 확인

---

## Day 2 — CAE 특화 ML 핵심 전략 3개
- [ ] Regression mapping 개념 이해
- [ ] Autoencoder 기반 차원 축소 개념 이해
- [ ] GNN(Graph Neural Network)의 mesh 표현 방식 이해
### 실습
- [ ] scikit-learn으로 간단 회귀 모델 실행

---

## Day 3 — Geometry 기반 ML(Neural Concept의 핵심)
- [ ] CAD → ML 전체 pipeline 이해
- [ ] Geometry representation(voxel/point/SDF) 차이 이해
- [ ] Geometry parameterization 기법 이해
- [ ] Geometry surrogate 개념 설명 가능
### 실습
- [ ] CAD parameter → scalar 테이블 수동 생성(10개 이상)

---

## Day 4 — CAE 데이터 전처리 → ML 입력
- [ ] Geometry normalization 방식 이해
- [ ] Scalar field normalization 이해
- [ ] Mesh mismatch 해결(interpolation/resampling)
- [ ] Dataset input→output 매핑 정의
### 실습
- [ ] pyvista로 field 추출 후 CSV로 저장

---

## Day 5 — Surrogate Model 기본 구조(MLP)
- [ ] MLP 구조 이해(Linear + ReLU)
- [ ] Loss function 의미(MSE/MAE)
- [ ] Overfitting 개념과 방지 전략
- [ ] Train/validation split 필요성 이해
### 실습
- [ ] PyTorch로 20~30줄짜리 MLP 직접 구현

---

## Day 6 — CAE 의미 기반 모델 평가
- [ ] CAE 특화 metric(max stress error 등)
- [ ] Physics consistency 개념
- [ ] Engineering sanity check 원리
### 실습
- [ ] 예측 vs 실측 그래프 생성 및 오차 분석

---

## Day 7 — End-to-End Mini Project
- [ ] Beam bending dataset 생성
- [ ] PyTorch surrogate 학습
- [ ] 성능 평가 + 그래프 분석
- [ ] 전체 CAE→ML pipeline 스스로 설명 가능
### 실습
- [ ] dataset.csv 생성
- [ ] train.py 실행
- [ ] loss/line/scatter plot 생성


# 30일 심화 학습 체크리스트  
### Machine Learning in Modeling and Simulation (Rabczuk & Bathe)

---

## Ch1 — Machine Learning in CAE (Day 1–3)
- [ ] CAE에서 ML 필요성 정리
- [ ] Simulation-based dataset 특성 이해
- [ ] Surrogate workflow 정리
- [ ] Offline surrogate vs Online digital twin 차이
### 실습
- [ ] CAE case 10개 생성 및 feature/target 정의

---

## Ch2 — Artificial Neural Networks (Day 4–6)
- [ ] MLP 구조 이해
- [ ] Overfitting 방지 기법 정리
- [ ] Activation function 비교(ReLU/tanh/GELU)
- [ ] Loss landscape 개념 이해
### 실습
- [ ] dropout/hidden layer 변경 후 성능 비교

---

## Ch3 — Gaussian Processes (Day 7–9)
- [ ] Kernel function(RBF/Matérn) 개념 이해
- [ ] GP posterior 계산 원리
- [ ] GP 장점/단점 정리
- [ ] High-dimensional 문제에서 발생하는 계산비용 정리
### 실습
- [ ] RBF vs Matérn kernel 비교 실험

---

## Ch4 — ML for Dynamic Models From Data (Day 10–12)
- [ ] State-space 모델 개념
- [ ] SINDy 알고리즘 개념
- [ ] LSTM/GRU 구조 이해
- [ ] Dynamic system ID vs black-box 비교
### 실습
- [ ] 2-DOF 시스템 + LSTM 예측 실험

---

## Ch5 — Physics-Informed Neural Networks (PINNs) (Day 13–15)
- [ ] PINN loss 구조(PDE + BC + data) 이해
- [ ] Automatic differentiation 개념
- [ ] Hard/Soft constraint 차이
- [ ] Multi-physics PINN 난제 정리
### 실습
- [ ] 1D 열전달 PINN 구현

---

## Ch6 — Physics-Informed Neural Operator Networks (Day 16–18)
- [ ] Neural Operator 개념
- [ ] FNO 구조 이해
- [ ] DeepONet(branch + trunk) 구조 이해
- [ ] PINN vs Neural Operator 비교
### 실습
- [ ] Poisson 문제 FNO 학습

---

## Ch7 — Digital Twin (Day 19–21)
- [ ] Digital twin 구성요소(모델 + 센서 + 업데이트)
- [ ] Kalman filter 기반 상태추정 개념
- [ ] Online learning 개념
- [ ] Sensor + simulation 통합 구조 이해
### 실습
- [ ] 노이즈 있는 센서 신호 + Kalman filter 적용

---

## Ch8 — Reduced Order Modeling (Day 22–24)
- [ ] POD 이해(SVD 기반)
- [ ] POD-Galerkin projection
- [ ] Hyper-reduction(DEIM 등) 개념
- [ ] Autoencoder ROM 비교
### 실습
- [ ] POD-ROM 구현 및 성능 향상 측정

---

## Ch9 — Regression Models (Day 25–27)
- [ ] Linear/Polynomial regression 이해
- [ ] Feature scaling 전략
- [ ] Regularization(Lasso/Ridge) 이해
- [ ] Bias–variance trade-off 개념
### 실습
- [ ] Polynomial 회귀 degree 2–8 비교 실험

---

## Ch10 — ML-Assisted Topology Optimization (Day 28–30)
- [ ] SIMP 기반 topology optimization 원리
- [ ] Density field → property 관계 이해
- [ ] CNN/U-Net 기반 generative design 개념
- [ ] Multi-objective ML+TO 구조 이해
### 실습
- [ ] 2D TO density dataset 구축 & surrogate 학습
