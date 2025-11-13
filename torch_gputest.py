import time
import torch


def run_matmul(device, n_iter=20):
    print(f"\n=== Device: {device} ===")
    size = 8000  # 큰 행렬

    x = torch.randn(size, size, device=device)
    w = torch.randn(size, size, device=device)

    def sync():
        # 디바이스 타입에 맞는 synchronize 호출
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        # cpu는 필요 없음

        # 다른 backend가 추가되면 여기에 elif로 확장 가능

    # 워밍업
    for _ in range(3):
        y = x @ w
        sync()

    start = time.perf_counter()
    for _ in range(n_iter):
        y = x @ w
        sync()
    end = time.perf_counter()

    print(f"Elapsed: {end - start:.3f} sec for {n_iter} matmuls")


def main():
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    print("CUDA available:", has_cuda)
    print("MPS available:", has_mps)

    # 항상 CPU 먼저 기준으로 한 번
    run_matmul(torch.device("cpu"))

    # 가능한 가속기에서 한 번 더 (우선순위: CUDA > MPS)
    if has_cuda:
        run_matmul(torch.device("cuda"))
    elif has_mps:
        run_matmul(torch.device("mps"))
    else:
        print("\nNo CUDA or MPS available. Only CPU test was run.")


if __name__ == "__main__":
    main()
