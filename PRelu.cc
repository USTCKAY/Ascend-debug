#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "common/logging.h"

#include "common/nputensor.h"

int main() {
  // Init
  ACL_CALL(aclInit(nullptr));
  ACL_CALL(aclrtSetDevice(0));

  // Get Run Mode - ACL_HOST
  aclrtRunMode runMode;
  ACL_CALL(aclrtGetRunMode(&runMode));
  std::string run_mode_str = (runMode == ACL_DEVICE) ? "ACL_DEVICE" : "ACL_HOST";
  std::cout << "aclrtRunMode is : " << run_mode_str << std::endl;

  // op type
  const std::string op_type = "PRelu";
  // input - x, y
  const std::vector<int64_t> x_dims{2, 3, 1, 1};
  const std::vector<int64_t> y_dims{3};
  const std::vector<float> x_data(6, -1.0);
  const std::vector<float> y_data{-1, -2, -3};
  // output
  std::vector<int64_t> out_dims{2, 3, 1, 1};

  // inputs
  auto input_x = new npuTensor<float>(ACL_FLOAT, x_dims.size(), x_dims.data(), ACL_FORMAT_NCHW, x_data.data());
  auto input_y = new npuTensor<float>(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW, y_data.data());
  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_x->desc);
  input_descs.emplace_back(input_y->desc);
  input_buffers.emplace_back(input_x->buffer);
  input_buffers.emplace_back(input_y->buffer);

  // output
  auto output = new npuTensor<float>(ACL_FLOAT, out_dims.size(), out_dims.data(), ACL_FORMAT_NCHW, nullptr);
  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output->desc);
  output_buffers.emplace_back(output->buffer);

  // attr
  auto attr = aclopCreateAttr();

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
  input_y->Print("y");
  output->Print("output");

  // destroy
  input_x->Destroy();
  input_y->Destroy();
  output->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}
