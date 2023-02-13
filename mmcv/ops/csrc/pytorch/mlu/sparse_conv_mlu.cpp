/*************************************************************************
 * Copyright (C) 2022 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "mlu_common_helper.h"
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"

#include <torch/script.h>
#include <vector>

template <unsigned NDim>
std::vector<torch::Tensor> GetIndicePairsForwardMLUKernelLauncher(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  std::cout << "GetIndicePairsForwardMLUKernelLauncher start." << std::endl;

	// The following code is copied from mmcv/ops/csrc/pytorch/cuda/spconv_ops_cuda.cu
	// to ensure the output is available for network train.
	// The outputs of this function have correct shape but wrong value.
  auto numAct = indices.size(0);
  auto kernelVolume = kernelSize[0];
  for (int i = 1; i < kernelSize.size(); ++i) {
    kernelVolume *= kernelSize[i];
  }

  auto outputVolume = outSpatialShape[0];
  for (int i = 1; i < outSpatialShape.size(); ++i) {
    outputVolume *= outSpatialShape[i];
  }
  torch::Tensor indicePairs =
      at::full({kernelVolume, 2, numAct}, -1,
                  indices.options().dtype(at::kInt));
  torch::Tensor indiceNum = at::zeros(
      {kernelVolume}, indices.options().dtype(at::kInt));
  
  std::cout << "GetIndicePairsForwardMLUKernelLauncher finish." << std::endl;
	return {indices, indicePairs, indiceNum}; // differ from cuda code
}

torch::Tensor IndiceConvForwardMLUKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor indicePairs,
    torch::Tensor indiceNum, int64_t numActOut, int64_t _inverse,
    int64_t _subM) {
  std::cout << "IndiceConvForwardMLUKernelLauncher start." << std::endl;
	int C = filters.dim() == 4 ?
				  filters.size(3) : filters.size(4);
	torch::Tensor output = at::zeros({numActOut, C}, features.options().dtype(at::kFloat));

  std::cout << "IndiceConvForwardMLUKernelLauncher finish." << std::endl;
	return output;
}

std::vector<torch::Tensor> IndiceConvBackwardMLUKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM) {
  std::cout << "IndiceConvBackwardMLUKernelLauncher start." << std::endl;
  auto indice_num_cpu = indiceNum.to({torch::kCPU});
	auto indice_num_cpu_64 = indice_num_cpu.data_ptr<int>();
	int indice_num_len = indiceNum.numel();
	int64_t indice_num[indice_num_len];
	for (int i = 0; i < indice_num_len; ++i) {
    // indice_num[i] = ((int64_t *)(indice_num_cpu_64.unsafeGetTensorImpl()->data()))[i];
    indice_num[i] = (int64_t)(((int *)(indice_num_cpu_64))[i]);
    std::cout << "indice_num_cpu_64-" << i << " " << ((int *)(indice_num_cpu_64))[i] << std::endl;
    std::cout << "indice_num-" << i << " " << indice_num[i] << std::endl;
	}
	
	auto input_grad_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      features, features.suggest_memory_format());
  auto output_grad_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      outGrad, outGrad.suggest_memory_format());
  auto filters_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      filters, filters.suggest_memory_format());
  auto indice_pairs_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      indicePairs, indicePairs.suggest_memory_format());

  MluOpTensorDescriptor output_grad_desc, filters_desc, indice_pairs_desc, input_grad_desc;
  input_grad_desc.set(input_grad_contiguous);
  output_grad_desc.set(output_grad_contiguous);
  filters_desc.set(filters_contiguous);
  indice_pairs_desc.set(indice_pairs_contiguous);

  // need to set desc layout with mluOp functions
  {
    mluOpTensorLayout_t layout;
    mluOpDataType_t dtype;
    int dim;
    int dims[8];

    // output_grad_desc
    mluOpGetTensorDescriptor(output_grad_desc.desc(), &layout, &dtype, &dim, dims);
    mluOpSetTensorDescriptor(output_grad_desc.desc(), MLUOP_LAYOUT_ARRAY, dtype, dim, dims);

    // filters_desc
    mluOpGetTensorDescriptor(filters_desc.desc(), &layout, &dtype, &dim, dims);
    if (dim == 4) {
      mluOpSetTensorDescriptor(filters_desc.desc(), MLUOP_LAYOUT_HWCN, dtype, dim, dims);
    } else {
      mluOpSetTensorDescriptor(filters_desc.desc(), MLUOP_LAYOUT_ARRAY, dtype, dim, dims);
    }

    // indice_pairs_desc
    mluOpGetTensorDescriptor(indice_pairs_desc.desc(), &layout, &dtype, &dim, dims);
    mluOpSetTensorDescriptor(indice_pairs_desc.desc(), MLUOP_LAYOUT_ARRAY, dtype, dim, dims);

    // input_grad_desc
    mluOpGetTensorDescriptor(input_grad_desc.desc(), &layout, &dtype, &dim, dims);
    mluOpSetTensorDescriptor(input_grad_desc.desc(), MLUOP_LAYOUT_ARRAY, dtype, dim, dims);
  }

  auto handle = mluOpGetCurrentHandle();
	size_t workspace_size = 0;
  mluOpGetIndiceConvolutionBackwardDataWorkspaceSize(
			handle, output_grad_desc.desc(), filters_desc.desc(),
      indice_pairs_desc.desc(), input_grad_desc.desc(),
      indice_num, _inverse, &workspace_size);
	printf("mluOpGetIndiceConvolutionBackwardDataWorkspaceSize %ld\n", workspace_size);
  
	// generate empty input_grad
	torch::Tensor input_grad = at::zeros({features.size(0), features.size(1)}, features.options().dtype(at::kFloat));
	torch::Tensor filters_grad;
	if (filters.dim() == 4) {
		int h = filters.size(0);
		int w = filters.size(1);
		int c = filters.size(2);
		int n = filters.size(3);
		filters_grad = at::zeros({h, w, c, n}, filters.options().dtype(at::kFloat));
	} else if (filters.dim() == 5) {
		int d = filters.size(0);
		int h = filters.size(1);
		int w = filters.size(2);
		int c = filters.size(3);
		int n = filters.size(4);
		filters_grad = at::zeros({d, h, w, c, n}, filters.options().dtype(at::kFloat));
	}

  auto indice_convbpdata_workspace = at::empty(workspace_size, features.options().dtype(at::kByte));

	auto output_grad_impl = torch_mlu::getMluTensorImpl(output_grad_contiguous);
	auto filters_impl = torch_mlu::getMluTensorImpl(filters_contiguous);
	auto indice_pairs_impl = torch_mlu::getMluTensorImpl(indice_pairs_contiguous);
  auto indice_convbpdata_workspace_impl = torch_mlu::getMluTensorImpl(indice_convbpdata_workspace);

	auto output_grad_ptr = output_grad_impl->cnnlMalloc();
	auto filters_ptr = filters_impl->cnnlMalloc();
	auto indice_pairs_ptr = indice_pairs_impl->cnnlMalloc();
  auto indice_convbpdata_workspace_ptr = indice_convbpdata_workspace_impl->cnnlMalloc();

	auto input_grad_impl = torch_mlu::getMluTensorImpl(input_grad);
	auto input_grad_ptr = input_grad_impl->cnnlMalloc();

	mluOpIndiceConvolutionBackwardData(
			handle, output_grad_desc.desc(), output_grad_ptr, filters_desc.desc(), filters_ptr,
      indice_pairs_desc.desc(), indice_pairs_ptr, indice_num, _inverse, _subM,
      indice_convbpdata_workspace_ptr, workspace_size, input_grad_desc.desc(), input_grad_ptr);

	std::vector<torch::Tensor> result;
	result.push_back(input_grad);
	result.push_back(filters_grad);
  std::cout << "IndiceConvBackwardMLUKernelLauncher finish." << std::endl;
  return result;
}

torch::Tensor indice_conv_forward_mlu(
    torch::Tensor features, torch::Tensor filters, torch::Tensor indicePairs,
    torch::Tensor indiceNum, int64_t numActOut, int64_t _inverse,
    int64_t _subM) {
  return IndiceConvForwardMLUKernelLauncher(
      features, filters, indicePairs, indiceNum, numActOut, _inverse, _subM);
}

std::vector<torch::Tensor> indice_conv_backward_mlu(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM) {
  return IndiceConvBackwardMLUKernelLauncher(
      features, filters, outGrad, indicePairs, indiceNum, _inverse, _subM);
}

torch::Tensor indice_conv_forward_impl(torch::Tensor features,
                                       torch::Tensor filters,
                                       torch::Tensor indicePairs,
                                       torch::Tensor indiceNum,
                                       int64_t numActOut, int64_t _inverse,
                                       int64_t _subM);

std::vector<torch::Tensor> indice_conv_backward_impl(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM);

REGISTER_DEVICE_IMPL(indice_conv_forward_impl, MLU, indice_conv_forward_mlu);
REGISTER_DEVICE_IMPL(indice_conv_backward_impl, MLU, indice_conv_backward_mlu);


template std::vector<torch::Tensor> GetIndicePairsForwardMLUKernelLauncher<2>(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template std::vector<torch::Tensor> GetIndicePairsForwardMLUKernelLauncher<3>(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template std::vector<torch::Tensor> GetIndicePairsForwardMLUKernelLauncher<4>(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);