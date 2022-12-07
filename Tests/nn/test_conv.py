import math
import unittest
import itertools
import warnings
from itertools import product

import torch
import torch.autograd.forward_ad as fwAD
import torch.backends.cudnn as cudnn
from torch.backends import mkldnn
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import SourceChangeWarning # Use SourceChangeWarning
from torch.testing._internal.common_dtype import floating_types_and, floating_and_complex_types_and
from torch.testing._internal.common_utils import run_tests, \
    skipIfRocmVersionLessThan, skipIfNotMiopenSuggestNHWC, TEST_SCIPY, TEST_WITH_ROCM, \
    download_file, parametrize as parametrize_test, subtest, \
    instantiate_parametrized_tests, set_default_dtype
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_CUDNN
from torch.testing._internal.common_nn import NNTestCase, _test_module_empty_input
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, skipCUDAIfNoCudnn, skipCUDAIfCudnnVersionLessThan, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIfRocmVersionLessThan, skipCUDAIfNotMiopenSuggestNHWC, \
    onlyNativeDeviceTypes, largeTensorTest, skipMeta, \
    disableMkldnn, skipCPUIfNoMkldnn, disablecuDNN, skipCUDAIfMiopen, skipCUDAIfNoMiopen

from torch.testing import make_tensor
from torch.testing._internal.common_utils import gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32

AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

if TEST_SCIPY:
    import scipy.signal
    import scipy.ndimage

class TestConvolutionNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True  

    # Conv BackPropogation Compatibility
    def test_conv_backcompat(self):
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            m = torch.load(path, encoding='utf-8')
        input = torch.randn((1, 1, 1, 1), dtype=torch.float)
        self.assertEqual(m(input).size(), (1, 1, 1, 1))

    # Invalid Conv1D
    def test_invalid_conv1d(self):
        for dtype in [torch.bfloat16, torch.float, torch.double, torch.cfloat, torch.cdouble]:
            module = nn.Conv1d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 
                                        r'Calculated padded input size per channel: \(4\). ' +
                                        r'Kernel size: \(10\). Kernel size can\'t be greater than actual input size'):
                module(input)
            
            # Negative Stride Check
            module = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=-1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4).to(dtype)
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)
            
    # Mismatch Shape Conv2d
    def test_mismatch_shape_conv2d(self):
        for dtype in (torch.float, torch.cfloat):
            x = torch.randn(1, 10, 1, 28, 28, dtype=dtype)
            w = torch.randn(6, 1, 5, 5, dtype=dtype)

            with self.assertRaisesRegex(RuntimeError,
                                        r'Expected 3D \(unbatched\) or 4D \(batched\) input to conv2d, but got ' +
                                        r'input of size: \[1, 10, 1, 28, 28\]'):
                F.conv2d(x, w)
    
    # Conv2d Discontigous Weight Check
    def test_conv2d_discontigous_weight(self):
        for dtype in (torch.float, torch.cfloat):
            x = torch.ones(64, 16, 16, 16, dtype=dtype)
            weight = torch.arange(0, 1.0, 1 / 2.0 ** 10).reshape(32, 16, 1, 2).to(dtype)[:, :, :, ::2]
            self.assertFalse(weight.is_contiguous())
            y = F.conv2d(x, weight, None)
            if mkldnn.is_available():
                # Disable MKLDNN, so either NNPACK or THCNN would be used
                with mkldnn.flags(enabled=False):
                    y_ = F.conv2d(x, weight, None)
                    self.assertEqual(y, y_)
            self.assertEqual(y.sum(), 4186112.)

if __name__ == '__main__':
    run_tests()