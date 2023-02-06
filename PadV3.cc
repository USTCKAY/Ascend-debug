#include <iostream>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <numeric>
#include <unistd.h>

#include "common/nputensor.h"

int main() {
  std::cout << "current pid is: " << getpid() << std::endl;  
  // Init
  ACL_CALL(aclInit(nullptr));
  ACL_CALL(aclrtSetDevice(0));

  // Get Run Mode - ACL_HOST
  aclrtRunMode runMode;
  ACL_CALL(aclrtGetRunMode(&runMode));
  std::string run_mode_str = (runMode == ACL_DEVICE) ? "ACL_DEVICE" : "ACL_HOST";
  std::cout << "aclrtRunMode is : " << run_mode_str << std::endl;

  // op type
  const std::string op_type = "PadV3";
  // input - x
  const std::vector<int64_t> x_dims{1, 1, 1, 2, 3};
  const std::vector<float> x_data{0.701469, 0.286139, 0.226851, 0.551315, 0.719469, 0.423106};
  // const std::vector<int64_t> x_data{1, 2, 3, 4, 5, 6};
  // input - paddings
  const std::vector<int64_t> paddings_dims{10};
  const std::vector<int64_t> paddings_data{0, 0, 0, 0, 0, 0, 1, 1, 1, 0};
  // input-constant_value
  const std::vector<int64_t> constant_dims{1};
  // const std::vector<int64_t> constant_data{0};
  const std::vector<float> constant_data{0.0};
  // output
  const std::vector<int64_t> output_dims{1, 1, 1, 4, 4};

  // input 
  auto input_x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_ND, x_data.data(), memType::DEVICE);
  // auto input_x = new npuTensor<int64_t>(ACL_INT64, x_dims.size(), x_dims.data(), ACL_FORMAT_ND, x_data.data(), memType::DEVICE);
  auto input_paddings = new npuTensor<int64_t>(ACL_INT64, paddings_dims.size(), paddings_dims.data(), ACL_FORMAT_ND, paddings_data.data(), memType::HOST);
  auto input_constant = new npuTensor<float>(ACL_FLOAT, constant_dims.size(), constant_dims.data(), ACL_FORMAT_ND, constant_data.data(), memType::HOST);
  // auto input_constant = new npuTensor<int64_t>(ACL_INT64, constant_dims.size(), constant_dims.data(), ACL_FORMAT_ND, constant_data.data(), memType::HOST);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_descs.emplace_back(input_paddings->desc);
  input_descs.emplace_back(input_constant->desc);
  input_buffers.emplace_back(input_x->buffer);
  input_buffers.emplace_back(input_paddings->buffer);
  input_buffers.emplace_back(input_constant->buffer);

  // output - out
  auto output = new npuTensor<float>(ACL_FLOAT, output_dims.size(), output_dims.data(), ACL_FORMAT_ND, nullptr);
  // auto output = new npuTensor<int64_t>(ACL_INT64, output_dims.size(), output_dims.data(), ACL_FORMAT_ND, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output->desc);
  output_buffers.emplace_back(output->buffer);

  // attr
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrString(attr, "mode", "edge"));
  ACL_CALL(aclopSetAttrBool(attr, "paddings_contiguous", true));
  
  // create stream
  aclrtStream stream = nullptr;
  ACL_CALL(aclrtCreateStream(&stream));

  std::cout << "aclopCompileAndExecute : " << op_type << std::endl;
  ACL_CALL(aclopCompileAndExecute(op_type.c_str(), 
            input_descs.size(), input_descs.data(), input_buffers.data(), 
            output_descs.size(), output_descs.data(), output_buffers.data(), 
            attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));

  // sync and destroy stream
  ACL_CALL(aclrtSynchronizeStream(stream));
  ACL_CALL(aclrtDestroyStream(stream));

  // print output
  input_x->Print("x");
  input_paddings->Print("paddings");
  input_constant->Print("constant_values");
  output->Print("y");

  // destroy - outputs
  input_x->Destroy();
  input_paddings->Destroy();
  input_constant->Destroy();
  output->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}
