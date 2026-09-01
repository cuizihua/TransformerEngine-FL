#!/usr/bin/env python3
# Copyright (c) 2026, BAAI. All rights reserved.
#
# Test LayerNorm NPU Backend Implementation

"""
单元测试和性能对比测试 for LayerNorm NPU Backend

运行方式：
    python tests/plugin/backend/npu/test_layernorm_npu.py
    或：bash tests/plugin/backend/npu/run_layernorm_test.sh

测试内容：
1. 功能正确性测试（forward/backward）
2. 不同形状输入测试
3. zero_centered_gamma 测试
4. 性能对比测试（NPU vs PyTorch 原生）
"""

import torch
import torch_npu
import time
import sys
from pathlib import Path
from typing import Tuple

# 导入 FlagOS NPU Backend
# 从 tests/plugin/backend/npu/ 向上 4 层定位到仓库根目录，便于未安装时直接运行
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from transformer_engine.plugin.core.backends.vendor.npu.npu import NPUBackend


class TestLayerNormNPU:
    """LayerNorm NPU Backend 测试类"""

    def __init__(self):
        self.backend = NPUBackend()
        self.device = "npu:0"
        self.test_results = []

    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """记录测试结果"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append((test_name, passed, message))
        print(f"{status} | {test_name}")
        if message:
            print(f"         {message}")

    def test_basic_forward(self):
        """测试基本的 forward 功能"""
        test_name = "Basic Forward"
        try:
            batch_size, seq_len, hidden_size = 2, 4, 768

            # 准备输入
            x = torch.randn(
                batch_size, seq_len, hidden_size, device=self.device, dtype=torch.bfloat16
            )
            weight = torch.ones(hidden_size, device=self.device, dtype=torch.bfloat16)
            bias = torch.zeros(hidden_size, device=self.device, dtype=torch.bfloat16)
            eps = 1e-5

            # 调用 NPU backend
            out, mean, rstd = self.backend.layernorm_fwd(
                input=x,
                weight=weight,
                bias=bias,
                eps=eps,
                ln_out=None,
                quantizer=None,
                otype=None,
                sm_margin=0,
                zero_centered_gamma=False,
            )

            # 验证形状
            assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
            assert mean.shape[0] == batch_size * seq_len, f"Mean shape mismatch: {mean.shape}"
            assert rstd.shape[0] == batch_size * seq_len, f"Rstd shape mismatch: {rstd.shape}"

            # 验证输出不是 NaN 或 Inf
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"

            self.log_result(test_name, True, f"Shape: {x.shape} -> {out.shape}")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_basic_backward(self):
        """测试基本的 backward 功能"""
        test_name = "Basic Backward"
        try:
            batch_size, seq_len, hidden_size = 2, 4, 768

            # Forward pass
            x = torch.randn(
                batch_size, seq_len, hidden_size, device=self.device, dtype=torch.bfloat16
            )
            weight = torch.ones(hidden_size, device=self.device, dtype=torch.bfloat16)
            bias = torch.zeros(hidden_size, device=self.device, dtype=torch.bfloat16)
            eps = 1e-5

            out, mean, rstd = self.backend.layernorm_fwd(
                x, weight, bias, eps, None, None, None, 0, False
            )

            # Backward pass
            grad_out = torch.randn_like(out)
            dx, dgamma, dbeta = self.backend.layernorm_bwd(
                dz=grad_out,
                x=x,
                mu=mean,
                rsigma=rstd,
                gamma=weight,
                sm_margin=0,
                zero_centered_gamma=False,
            )

            # 验证形状
            assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape} vs {x.shape}"
            assert (
                dgamma.shape == weight.shape
            ), f"dgamma shape mismatch: {dgamma.shape} vs {weight.shape}"
            assert dbeta.shape == bias.shape, f"dbeta shape mismatch: {dbeta.shape} vs {bias.shape}"

            # 验证梯度不是 NaN 或 Inf
            assert not torch.isnan(dx).any(), "dx contains NaN"
            assert not torch.isnan(dgamma).any(), "dgamma contains NaN"
            assert not torch.isnan(dbeta).any(), "dbeta contains NaN"

            self.log_result(
                test_name,
                True,
                f"Gradients: dx{dx.shape}, dgamma{dgamma.shape}, dbeta{dbeta.shape}",
            )

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_zero_centered_gamma(self):
        """测试 zero_centered_gamma 参数"""
        test_name = "Zero-Centered Gamma"
        try:
            batch_size, seq_len, hidden_size = 2, 4, 768

            x = torch.randn(
                batch_size, seq_len, hidden_size, device=self.device, dtype=torch.bfloat16
            )
            weight = torch.zeros(hidden_size, device=self.device, dtype=torch.bfloat16)  # 初始化为0
            bias = torch.zeros(hidden_size, device=self.device, dtype=torch.bfloat16)
            eps = 1e-5

            # 测试 zero_centered_gamma=True
            out_zcg, mean_zcg, rstd_zcg = self.backend.layernorm_fwd(
                x, weight, bias, eps, None, None, None, 0, zero_centered_gamma=True
            )

            # 测试 zero_centered_gamma=False with weight+1
            weight_plus_one = weight + 1
            out_normal, mean_normal, rstd_normal = self.backend.layernorm_fwd(
                x, weight_plus_one, bias, eps, None, None, None, 0, zero_centered_gamma=False
            )

            # 两者应该给出相同的结果
            diff = (out_zcg - out_normal).abs().max().item()

            self.log_result(
                test_name, True, f"Max diff between zero_centered and normal: {diff:.6f}"
            )

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_different_shapes(self):
        """测试不同输入形状"""
        test_name = "Different Input Shapes"
        try:
            shapes = [
                (2, 128, 768),  # 小 batch
                (8, 512, 1024),  # 中等大小
                (16, 2048, 4096),  # 大尺寸
                (1, 1, 768),  # 最小
            ]

            passed = True
            for shape in shapes:
                batch_size, seq_len, hidden_size = shape
                x = torch.randn(
                    batch_size, seq_len, hidden_size, device=self.device, dtype=torch.bfloat16
                )
                weight = torch.ones(hidden_size, device=self.device, dtype=torch.bfloat16)
                bias = torch.zeros(hidden_size, device=self.device, dtype=torch.bfloat16)

                out, mean, rstd = self.backend.layernorm_fwd(
                    x, weight, bias, 1e-5, None, None, None, 0, False
                )

                if out.shape != x.shape:
                    passed = False
                    break

            self.log_result(test_name, passed, f"Tested {len(shapes)} different shapes")

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_numerical_correctness(self):
        """测试数值正确性（与 PyTorch 原生对比）"""
        test_name = "Numerical Correctness vs PyTorch"
        try:
            batch_size, seq_len, hidden_size = 4, 64, 768

            # 准备相同的输入
            x = torch.randn(
                batch_size, seq_len, hidden_size, device=self.device, dtype=torch.float32
            )
            weight = torch.randn(hidden_size, device=self.device, dtype=torch.float32)
            bias = torch.randn(hidden_size, device=self.device, dtype=torch.float32)
            eps = 1e-5

            # NPU Backend
            out_npu, _, _ = self.backend.layernorm_fwd(
                x, weight, bias, eps, None, None, None, 0, False
            )

            # PyTorch 原生
            out_torch = torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias, eps)

            # 计算误差
            abs_diff = (out_npu - out_torch).abs()
            max_diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()

            # 判断是否通过（允许一定的数值误差）
            passed = max_diff < 1e-3

            self.log_result(
                test_name, passed, f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}"
            )

        except Exception as e:
            self.log_result(test_name, False, str(e))

    def test_gradient_correctness(self):
        """测试梯度正确性（与 PyTorch autograd 对比）"""
        test_name = "Gradient Correctness vs PyTorch Autograd"
        try:
            batch_size, seq_len, hidden_size = 4, 64, 768

            # 准备输入（需要 requires_grad）
            x_npu = torch.randn(
                batch_size, seq_len, hidden_size, device=self.device, dtype=torch.float32
            )
            weight_npu = torch.randn(hidden_size, device=self.device, dtype=torch.float32)
            bias_npu = torch.randn(hidden_size, device=self.device, dtype=torch.float32)

            x_torch = x_npu.clone().requires_grad_(True)
            weight_torch = weight_npu.clone().requires_grad_(True)
            bias_torch = bias_npu.clone().requires_grad_(True)

            eps = 1e-5

            # NPU Backend forward
            out_npu, mean_npu, rstd_npu = self.backend.layernorm_fwd(
                x_npu, weight_npu, bias_npu, eps, None, None, None, 0, False
            )

            # PyTorch autograd forward
            out_torch = torch.nn.functional.layer_norm(
                x_torch, (hidden_size,), weight_torch, bias_torch, eps
            )

            # 相同的上游梯度
            grad_out = torch.randn_like(out_npu)

            # NPU Backend backward
            dx_npu, dgamma_npu, dbeta_npu = self.backend.layernorm_bwd(
                grad_out, x_npu, mean_npu, rstd_npu, weight_npu, 0, False
            )

            # PyTorch autograd backward
            out_torch.backward(grad_out)

            # 比较梯度
            dx_diff = (dx_npu - x_torch.grad).abs().max().item()
            dgamma_diff = (dgamma_npu - weight_torch.grad).abs().max().item()
            dbeta_diff = (dbeta_npu - bias_torch.grad).abs().max().item()

            # 判断是否通过
            passed = dx_diff < 1e-3 and dgamma_diff < 1e-3 and dbeta_diff < 1e-3

            self.log_result(
                test_name,
                passed,
                f"dx diff: {dx_diff:.6e}, dgamma diff: {dgamma_diff:.6e}, dbeta diff:"
                f" {dbeta_diff:.6e}",
            )

        except Exception as e:
            self.log_result(test_name, False, str(e))


class PerformanceBenchmark:
    """性能对比测试"""

    def __init__(self):
        self.backend = NPUBackend()
        self.device = "npu:0"

    def benchmark_forward(
        self, batch_size: int, seq_len: int, hidden_size: int, num_iterations: int = 100
    ):
        """Benchmark forward pass"""
        print(f"\n{'='*70}")
        print(f"Forward Performance Benchmark: [{batch_size}, {seq_len}, {hidden_size}]")
        print(f"{'='*70}")

        # 准备输入
        x = torch.randn(batch_size, seq_len, hidden_size, device=self.device, dtype=torch.bfloat16)
        weight = torch.randn(hidden_size, device=self.device, dtype=torch.bfloat16)
        bias = torch.randn(hidden_size, device=self.device, dtype=torch.bfloat16)
        eps = 1e-5

        # Warmup
        for _ in range(10):
            _ = self.backend.layernorm_fwd(x, weight, bias, eps, None, None, None, 0, False)
            _ = torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias, eps)

        torch_npu.npu.synchronize()

        # Benchmark NPU Backend
        start = time.time()
        for _ in range(num_iterations):
            out_npu, _, _ = self.backend.layernorm_fwd(
                x, weight, bias, eps, None, None, None, 0, False
            )
        torch_npu.npu.synchronize()
        time_npu = (time.time() - start) / num_iterations * 1000  # ms

        # Benchmark PyTorch Native
        start = time.time()
        for _ in range(num_iterations):
            out_torch = torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias, eps)
        torch_npu.npu.synchronize()
        time_torch = (time.time() - start) / num_iterations * 1000  # ms

        # 计算加速比
        speedup = time_torch / time_npu

        print(f"NPU Backend:     {time_npu:.4f} ms")
        print(f"PyTorch Native:  {time_torch:.4f} ms")
        print(f"Speedup:         {speedup:.2f}x {'🚀' if speedup > 1 else '⚠️'}")

        return time_npu, time_torch, speedup

    def benchmark_backward(
        self, batch_size: int, seq_len: int, hidden_size: int, num_iterations: int = 100
    ):
        """Benchmark backward pass"""
        print(f"\n{'='*70}")
        print(f"Backward Performance Benchmark: [{batch_size}, {seq_len}, {hidden_size}]")
        print(f"{'='*70}")

        # 准备输入
        x = torch.randn(batch_size, seq_len, hidden_size, device=self.device, dtype=torch.bfloat16)
        weight = torch.randn(hidden_size, device=self.device, dtype=torch.bfloat16)
        bias = torch.randn(hidden_size, device=self.device, dtype=torch.bfloat16)
        eps = 1e-5

        # Forward pass for both
        out_npu, mean_npu, rstd_npu = self.backend.layernorm_fwd(
            x, weight, bias, eps, None, None, None, 0, False
        )
        grad_out = torch.randn_like(out_npu)

        # PyTorch setup
        x_torch = x.clone().requires_grad_(True)
        weight_torch = weight.clone().requires_grad_(True)
        bias_torch = bias.clone().requires_grad_(True)

        # Warmup
        for _ in range(10):
            _ = self.backend.layernorm_bwd(grad_out, x, mean_npu, rstd_npu, weight, 0, False)
            out_torch = torch.nn.functional.layer_norm(
                x_torch, (hidden_size,), weight_torch, bias_torch, eps
            )
            out_torch.backward(grad_out, retain_graph=True)

        torch_npu.npu.synchronize()

        # Benchmark NPU Backend
        start = time.time()
        for _ in range(num_iterations):
            dx_npu, dgamma_npu, dbeta_npu = self.backend.layernorm_bwd(
                grad_out, x, mean_npu, rstd_npu, weight, 0, False
            )
        torch_npu.npu.synchronize()
        time_npu = (time.time() - start) / num_iterations * 1000  # ms

        # Benchmark PyTorch Native
        start = time.time()
        for _ in range(num_iterations):
            x_torch.grad = None
            weight_torch.grad = None
            bias_torch.grad = None
            out_torch = torch.nn.functional.layer_norm(
                x_torch, (hidden_size,), weight_torch, bias_torch, eps
            )
            out_torch.backward(grad_out, retain_graph=True)
        torch_npu.npu.synchronize()
        time_torch = (time.time() - start) / num_iterations * 1000  # ms

        # 计算加速比
        speedup = time_torch / time_npu

        print(f"NPU Backend:     {time_npu:.4f} ms")
        print(f"PyTorch Native:  {time_torch:.4f} ms")
        print(f"Speedup:         {speedup:.2f}x {'🚀' if speedup > 1 else '⚠️'}")

        return time_npu, time_torch, speedup

    def run_comprehensive_benchmark(self):
        """运行全面的性能对比测试"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("=" * 70)

        test_configs = [
            # (batch_size, seq_len, hidden_size, description)
            (2, 128, 768, "Small (GPT-2 Small)"),
            (4, 512, 1024, "Medium (GPT-2 Medium)"),
            (8, 1024, 1536, "Large (GPT-2 Large)"),
            (16, 2048, 2048, "XLarge"),
        ]

        results = []

        for batch_size, seq_len, hidden_size, desc in test_configs:
            print(f"\n📊 Config: {desc}")
            print(f"   Shape: [{batch_size}, {seq_len}, {hidden_size}]")

            # Forward
            fwd_npu, fwd_torch, fwd_speedup = self.benchmark_forward(
                batch_size, seq_len, hidden_size, num_iterations=50
            )

            # Backward
            bwd_npu, bwd_torch, bwd_speedup = self.benchmark_backward(
                batch_size, seq_len, hidden_size, num_iterations=50
            )

            results.append(
                {
                    "config": desc,
                    "shape": (batch_size, seq_len, hidden_size),
                    "fwd_npu": fwd_npu,
                    "fwd_torch": fwd_torch,
                    "fwd_speedup": fwd_speedup,
                    "bwd_npu": bwd_npu,
                    "bwd_torch": bwd_torch,
                    "bwd_speedup": bwd_speedup,
                }
            )

        # 打印汇总表格
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Config':<20} {'Forward Speedup':<18} {'Backward Speedup':<18}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['config']:<20} {r['fwd_speedup']:>8.2f}x"
                f" {'🚀' if r['fwd_speedup'] > 1 else '⚠️':>8} {r['bwd_speedup']:>8.2f}x"
                f" {'🚀' if r['bwd_speedup'] > 1 else '⚠️':>8}"
            )
        print("=" * 70)


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("LayerNorm NPU Backend Test Suite")
    print("=" * 70)

    # 检查 NPU 可用性
    if not torch.npu.is_available():
        print("❌ NPU not available. Please run on NPU device.")
        return

    print(f"✅ NPU is available: {torch.npu.get_device_name(0)}")
    print(f"✅ torch_npu version: {torch_npu.__version__}")

    # 1. 运行功能测试
    print("\n" + "=" * 70)
    print("FUNCTIONAL TESTS")
    print("=" * 70)

    tester = TestLayerNormNPU()
    tester.test_basic_forward()
    tester.test_basic_backward()
    tester.test_zero_centered_gamma()
    tester.test_different_shapes()
    tester.test_numerical_correctness()
    tester.test_gradient_correctness()

    # 打印测试总结
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    total_tests = len(tester.test_results)
    passed_tests = sum(1 for _, passed, _ in tester.test_results if passed)
    print(f"Total: {total_tests} tests")
    print(f"Passed: {passed_tests} tests")
    print(f"Failed: {total_tests - passed_tests} tests")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")

    # 2. 运行性能测试
    if passed_tests == total_tests:
        print("\n✅ All functional tests passed! Running performance benchmarks...")
        benchmark = PerformanceBenchmark()
        benchmark.run_comprehensive_benchmark()
    else:
        print("\n⚠️ Some functional tests failed. Skipping performance benchmarks.")
        print("\nFailed tests:")
        for name, passed, message in tester.test_results:
            if not passed:
                print(f"  ❌ {name}: {message}")


if __name__ == "__main__":
    main()
