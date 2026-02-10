# RTS: RL-based Takt-time Scheduling Optimizer

이 프로젝트는 강화학습(Reinforcement Learning)을 활용하여 공장의 제품별/공정별 장비 할당을 최적화하는 시스템입니다. 생산 목표(Plan) 달성율을 극대화하고 장비 활용도를 최적화하는 것을 목표로 합니다.

## 🚀 주요 특징

### 1. 하이브리드 데이터 매니지먼트
- **학습(Training)**: 대량의 시나리오 데이터를 효율적으로 처리하기 위해 로컬 **JSON 파일** 기반으로 동작합니다.
- **추론(Inference)**: **Oracle DB**와 직접 연동하여 실시간 데이터를 읽고 시뮬레이션 결과를 다시 DB에 적재할 수 있습니다.
- **데이터 동기화**: `sync-db` 명령어를 통해 DB 데이터를 학습용 JSON 시나리오로 즉시 변환 가능합니다.

### 2. 표준화된 시뮬레이션 환경 (ProductionEnv)
- **가변 차원 대응**: 시나리오마다 다른 제품/공정 수를 고정된 5x5 그리드로 표준화(Padding)하여 하나의 모델이 다양한 환경에 대응할 수 있습니다.
- **정교한 물리 로직**:
  - 장비 이동 시 설정된 **체인지오버(Changeover)** 시간 동안 생산 중단 반영.
  - 장비 **다운타임(Downtime)** 스케줄에 따른 실시간 가용량 감소 시뮬레이션.
  - 각 장비 모델별 생산 효율(ST) 차이 반영.

### 3. 멀티 모드 추론
- **RL Mode**: 학습된 PPO 모델을 사용하여 지능적인 장비 할당 수행.
- **Heuristic Mode**: 전문가 규칙(WIP 우선 등) 기반의 베이스라인 알고리즘 제공.

---

## 🛠 설치 및 설정

### 1. 필수 라이브러리
```bash
pip install gymnasium numpy pandas stable-baselines3 torch oracledb pydantic pyyaml
```

### 2. 환경 설정 (`config.yaml`)
`config.yaml` 파일에서 학습 파라미터 및 DB 접속 정보를 관리합니다.
- `db.enabled`: DB 연동 여부 (true/false)
- `db.dsn`: Oracle DB 접속 정보 (`host:port/service_name`)
- `env.max_prods / max_procs`: 시뮬레이션 최대 차원 설정 (기본 5x5)

### 3. 데이터베이스 초기화
[oracle_setup.sql](file:///c:/Users/jaehw/Desktop/개발/rts/oracle_setup.sql) 파일을 실행하여 필요한 테이블(`RTS_**_INF`)을 생성합니다.

---

## 📖 사용 방법

### 1. 학습 (Training)
`data/` 폴더 내의 모든 시나리오를 사용하여 모델을 학습시킵니다.
```bash
python main.py train
```

### 2. 추론 (Inference)

#### **DB 기반 실시간 추론**
특정 `RULE_TIMEKEY` 값을 지정하여 DB에서 데이터를 읽고 결과를 적재합니다.
```bash
python main.py infer --mode rl --timekey 20260211000005
```

#### **로컬 파일 기반 추론**
특정 시나리오 폴더(예: `scn#5`)를 지정하여 실행합니다.
```bash
python main.py infer --mode rl --scenario scn#5
```

### 3. DB 데이터 동기화
DB의 특정 데이터를 학습용 JSON 시나리오 폴더로 생성합니다.
```bash
python main.py sync-db --timekey 20260211000005
```

---

## 📊 데이터베이스 테이블 구조 (Prefix: RTS_, Suffix: _INF)

- `RTS_EQP_CAPA_INF`: 제품/공정/모델별 생산 가능 여부 및 ST (INITIAL_COUNT 포함)
- `RTS_CO_RULE_INF`: 제품/공정 전환 시 발생하는 체인지오버 시간 설정
- `RTS_EQP_INV_INF`: 보유 장비 모델별 총 수량
- `RTS_PLAN_WIP_INF`: 현재 WIP 현황 및 생산 목표(Plan)
- `RTS_EQP_DT_INF`: 장비 모델별 고장/점검(Downtime) 스케줄
- `RTS_RESLT_INF`: 추론 결과(시뮬레이션 로그) 저장용 테이블

---

## 📂 프로젝트 구조 및 주요 파일 역할

### 1. 시각적 구조도
```text
rts/
├── main.py                 # 프로젝트 메인 엔트리 포인트 (CLI)
├── config.yaml             # 전체 설정 파일 (학습, DB, 로그 등)
├── oracle_setup.sql        # Oracle DB 테이블 생성 및 설정 SQL
├── all_scenarios_inserts.sql # 기존 샘플 데이터 DB 이관용 SQL
├── src/
│   └── rts/
│       ├── config/
│       │   └── config_manager.py # Pydantic 기반 설정 유효성 검사
│       ├── data/
│       │   ├── data_loader.py    # JSON 데이터 파싱 및 전처리
│       │   └── db_manager.py      # OracleDB 연동 및 결과 적재 로직
│       ├── env/
│       │   ├── factory_env.py    # 핵심 시뮬레이션 환경 (ProductionEnv)
│       │   └── __init__.py
│       ├── models/
│       │   ├── expert.py         # Heuristic 베이스라인 모델
│       │   ├── inference.py      # 추론 실행 및 결과 분석 파이프라인
│       │   ├── train.py          # 강화학습 모델 학습 로직
│       │   └── __init__.py
│       └── utils/
│           ├── logging_config.py # 중앙 집중형 로깅 설정
│           └── system_checker.py # 하드웨어/데이터 사전 검사 도구
├── data/                   # 학습용 JSON 시나리오 데이터셋 (scn#1 ~ scn#5)
└── logs/                   # 실행 로그 저장 디렉토리
```

### 2. 주요 파일 및 디렉토리 역할

| 분류 | 파일/디렉토리 | 주요 역할 |
| :--- | :--- | :--- |
| **Core** | `main.py` | 학습/추론/DB동기화 명령어를 통합 관리하는 실행 도구 |
| | `config.yaml` | 환경 변수, DB 접속 정보, 강화학습 파라미터 제어 |
| **Environment** | `factory_env.py` | 공정/제품/장비 간의 물리적 생산 로직 및 보상 함수 정의 (ProductionEnv) |
| **Data Layer** | `db_manager.py` | Oracle DB의 시나리오 데이터를 읽고 추론 결과를 기록 |
| | `data_loader.py` | 로컬 JSON 파일 및 DB 결과값을 환경에 맞게 로드 |
| **Models** | `train.py` | Stable-Baselines3를 활용한 PPO 모델 학습 및 저장 |
| | `inference.py` | 학습된 모델 또는 Heuristic Rule을 사용한 시뮬레이션 수행 |
| | `expert.py` | 데이터 기반의 Heuristic(Rule-base) 할당 알고리즘 구현 |
| **Database** | `oracle_setup.sql` | Oracle DB 환경 구축을 위한 DDL 및 기본 설정 |
| **Dataset** | `data/` | 다양한 제조 시나리오(장비 수, 제품 수 등)별 입력 데이터 |
| **Utils**| `system_checker.py` | 실행 전 GPU 가용성, 데이터 정합성 등을 자가 진단 |
