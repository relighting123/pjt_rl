# RTS - Reinforcement Training System for EQP Allocation Optimizer

RTS is a commercial-grade reinforcement learning framework for optimizing factory production schedules and EQP Allocation Optimizer.

## Installation

1. Clone the repository.
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Configuration

The system is configured via `config.yaml`. You can customize:
- Environment reward weights and penalties.
- RL hyperparameters (learning rate, batch size, etc.).
- Logging levels and directory.

## Usage

The system uses a unified entry point `main.py`.

### Training
To train the RL model on all available scenarios:
```bash
# 기본 학습 (데이터셋 전체 활용)
python main.py train --data_dir ./data --config config.yaml

# 특정 시나리오만 집중 학습
python main.py train --scenario scn#1 --data_dir ./data

# 데이터 검증만 수행 (Dry Run)
python main.py train --dry-run
```

### Inference
To run a simulation and generate results:

**RL Mode (학습된 모델 사용):**
```bash
# 특정 시나리오 추론
python main.py infer --mode rl --scenario scn#1 --model_path ppo_eqp_allocator

# 다른 모델 파일 지정
python main.py infer --mode rl --model_path models/my_final_model --scenario scn#2
```

**Heuristic Mode (규칙 기반 전문가):**
```bash
# 기본 휴리스틱 알고리즘 실행
python main.py infer --mode heuristic --scenario scn#1

# 결과 파일명 지정
python main.py infer --mode heuristic --scenario scn#2 --output results/scn2_heuristic.csv
```

## Data Structure
The `data/` directory should contain scenario folders (e.g., `scn#1`, `scn#2`), each with the following JSON files:
- `equipment_capability.json`: Product/Process/Model feasibility and ST.
- `changeover_rules.json`: Setup times between products.
- `equipment_inventory.json`: Available equipment counts per model.
- `plan_wip.json`: Production targets and initial WIP.

## Logging
Logs are saved in the `logs/` directory:
- `rts.log`: Detailed application logs.
- `error.log`: Runtime exception stack traces.
- `system_status.log`: CPU/Memory usage and system health.