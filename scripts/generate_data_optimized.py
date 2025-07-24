import argparse
import logging
import os
import gc
from typing import Dict, Any

import numpy as np
import orjson
from tqdm import tqdm

from rl4co.data.utils import check_extension
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def generate_data_in_chunks(
    problem: str,
    data_dir: str,
    dataset_size: int,
    graph_size: int,
    chunk_size: int = 1000,
    **kwargs,
) -> None:
    """메모리 효율적인 청킹 방식으로 데이터 생성"""

    num_chunks = (dataset_size + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(num_chunks), desc=f"Generating {problem} data"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, dataset_size)
        current_chunk_size = end_idx - start_idx

        log.info(
            f"Processing chunk {chunk_idx + 1}/{num_chunks} "
            f"(samples {start_idx}-{end_idx})"
        )

        # 청크별 데이터 생성
        chunk_data = generate_env_data_chunk(
            problem, data_dir, current_chunk_size, graph_size, **kwargs
        )

        # 청크 저장
        save_chunk_data(chunk_data, problem, graph_size, chunk_idx, **kwargs)

        # 메모리 정리
        del chunk_data
        gc.collect()


def generate_env_data_chunk(
    env_type: str, data_dir: str, chunk_size: int, graph_size: int, **kwargs
) -> Dict[str, np.ndarray]:
    """청크 단위로 환경 데이터 생성"""

    # 메모리 효율적인 샘플링
    sampled_data = sample_data_efficient(data_dir, chunk_size, graph_size, **kwargs)

    if env_type == "rcvrptw":
        return prepare_rcvrptw_data_chunk(sampled_data, chunk_size, graph_size)
    elif env_type == "rcvrp":
        return prepare_rcvrp_data_chunk(sampled_data, chunk_size, graph_size)
    elif env_type == "atsp":
        return prepare_atsp_data_chunk(sampled_data, chunk_size, graph_size)
    else:
        raise NotImplementedError(f"Environment type '{env_type}' not implemented.")


def sample_data_efficient(
    data_dir: str, chunk_size: int, graph_size: int, **kwargs
) -> Dict[str, np.ndarray]:
    """메모리 효율적인 데이터 샘플링"""

    # 메모리 매핑을 사용한 데이터 로딩
    cities_list = load_cities_list_efficient(data_dir, **kwargs)

    sampled_data = {}
    for city in cities_list[:5]:  # 도시 수 제한
        data_path = f"{data_dir}/{city}/{city}_data.npz"
        if os.path.exists(data_path):
            with np.load(data_path, mmap_mode="r") as data:
                # 필요한 부분만 메모리에 로드
                indices = np.random.choice(
                    len(data["points"]),
                    min(graph_size, len(data["points"])),
                    replace=False,
                )

                if not sampled_data:
                    sampled_data = {
                        "points": data["points"][indices],
                        "distance": data["distance"][indices][:, indices],
                        "duration": data["duration"][indices][:, indices],
                    }
                else:
                    # 기존 데이터와 병합 (메모리 효율적으로)
                    sampled_data["points"] = np.vstack(
                        [sampled_data["points"], data["points"][indices]]
                    )
                    # 거리/지속시간 행렬도 효율적으로 병합
                    # (실제 구현에서는 더 정교한 병합 로직 필요)

    return sampled_data


def prepare_rcvrptw_data_chunk(
    sampled_data: Dict[str, np.ndarray], chunk_size: int, graph_size: int
) -> Dict[str, np.ndarray]:
    """RCVRPTW 데이터 청크 준비 (메모리 효율적)"""

    # 정규화
    locs = normalize_points_efficient(sampled_data["points"])
    normalized_duration = normalize_duration_efficient(sampled_data["duration"])

    # MTVRP 데이터 생성 (청크 크기로 제한)
    data = generate_mtvrp_data_efficient(
        dataset_size=chunk_size,
        num_loc=graph_size,
        duration_matrix=normalized_duration,
        variant="VRPTW",
    )

    # 메모리 효율적인 업데이트
    data.update(
        {
            "locs": locs.astype(np.float32),
            "distance_matrix": sampled_data["distance"].astype(np.float32),
            "duration_matrix": normalized_duration.astype(np.float32),
        }
    )

    return data


def normalize_points_efficient(points: np.ndarray) -> np.ndarray:
    """메모리 효율적인 포인트 정규화"""
    points_min = np.min(points, axis=1, keepdims=True)
    points_max = np.max(points, axis=1, keepdims=True)
    return (points - points_min) / (points_max - points_min + 1e-8)


def normalize_duration_efficient(duration: np.ndarray) -> np.ndarray:
    """메모리 효율적인 지속시간 정규화"""
    duration_min = np.min(duration, axis=(1, 2), keepdims=True)
    duration_max = np.max(duration, axis=(1, 2), keepdims=True)
    denom = np.where(duration_max - duration_min == 0, 1, duration_max - duration_min)
    return (duration - duration_min) / denom


def generate_mtvrp_data_efficient(
    dataset_size: int,
    num_loc: int,
    duration_matrix: np.ndarray,
    variant: str = "VRPTW",
    **kwargs,
) -> Dict[str, np.ndarray]:
    """메모리 효율적인 MTVRP 데이터 생성"""

    # 기본 데이터 생성
    demand_linehaul = (
        np.random.randint(1, 10, (dataset_size, num_loc)).astype(np.float32) / 50.0
    )

    # 시간 윈도우 생성 (메모리 효율적)
    service_time = (0.15 + 0.03 * np.random.rand(dataset_size, num_loc)).astype(
        np.float32
    )
    tw_length = (0.18 + 0.02 * np.random.rand(dataset_size, num_loc)).astype(np.float32)

    # 지속시간 행렬 기반 시간 윈도우 계산
    d_0i = duration_matrix[:, 0, 1:]
    d_i0 = duration_matrix[:, 1:, 0]
    d_max = np.maximum(d_0i, d_i0)
    h_max = (4.6 - service_time - tw_length) / (d_max + 1e-6) - 1
    tw_start = d_0i + (h_max - 1) * d_max * np.random.rand(dataset_size, num_loc)
    tw_end = tw_start + tw_length

    # 시간 윈도우 배열 생성
    time_windows = np.concatenate(
        [np.zeros((dataset_size, 1, 2)), np.stack([tw_start, tw_end], axis=-1)], axis=1
    ).astype(np.float32)
    time_windows[:, 0, 1] = 4.6

    # 서비스 시간 패딩
    service_time = np.pad(service_time, ((0, 0), (1, 0))).astype(np.float32)

    return {
        "demand_linehaul": demand_linehaul,
        "time_windows": time_windows,
        "service_time": service_time,
        "vehicle_capacity": np.full((dataset_size, 1), 1.0, dtype=np.float32),
        "speed": np.full((dataset_size, 1), 1.0, dtype=np.float32),
    }


def save_chunk_data(
    chunk_data: Dict[str, np.ndarray],
    problem: str,
    graph_size: int,
    chunk_idx: int,
    **kwargs,
) -> None:
    """청크 데이터 저장"""

    save_dir = f"data/{problem}"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{problem}_n{graph_size}_chunk{chunk_idx:03d}.npz"
    filepath = os.path.join(save_dir, filename)

    log.info(f"Saving chunk {chunk_idx} to {filepath}")
    np.savez_compressed(filepath, **chunk_data)


def load_cities_list_efficient(data_dir: str, **kwargs) -> list:
    """메모리 효율적인 도시 목록 로딩"""
    data_path = os.path.join(data_dir, "splited_cities_list.json")
    with open(data_path, "r") as f:
        cities_list = orjson.loads(f.read())
        return cities_list.get("train", [])[:10]  # 도시 수 제한


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="rcvrptw")
    parser.add_argument("--dataset_size", type=int, default=1000)  # 작은 크기로 시작
    parser.add_argument("--graph_size", type=int, default=50)  # 작은 그래프로 시작
    parser.add_argument("--chunk_size", type=int, default=100)  # 청크 크기
    parser.add_argument("--data_dir", default="data/dataset")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 메모리 효율적인 데이터 생성
    generate_data_in_chunks(
        problem=args.problem,
        data_dir=args.data_dir,
        dataset_size=args.dataset_size,
        graph_size=args.graph_size,
        chunk_size=args.chunk_size,
    )
