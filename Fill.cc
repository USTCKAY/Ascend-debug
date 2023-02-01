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
  const std::string op_type = "Fill";
  // input - dims
  const std::vector<int64_t> dims_dims{3};
  const std::vector<int64_t> dims_data{2, 769, 769};
  // input - value
  const std::vector<int64_t> value_dims{1};
  const std::vector<float> value_data{1};
  // output
  const std::vector<int64_t> output_dims{2, 769, 769};

  // input - value
  auto input_dims = new npuTensor<int64_t>(ACL_INT64, dims_dims.size(), dims_dims.data(), ACL_FORMAT_ND, dims_data.data(), memType::HOST);
  auto input_value = new npuTensor<float>(ACL_FLOAT, value_dims.size(), value_dims.data(), ACL_FORMAT_ND, value_data.data(), memType::DEVICE);

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(input_dims->desc);
  input_descs.emplace_back(input_value->desc);
  input_buffers.emplace_back(input_dims->buffer);
  input_buffers.emplace_back(input_value->buffer);

  // output - out
  auto output = new npuTensor<float>(ACL_FLOAT, output_dims.size(), output_dims.data(), ACL_FORMAT_ND, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(output->desc);
  output_buffers.emplace_back(output->buffer);

  // Note: need to change data type first
  // int64_t input_value = static_cast<int64_t>(value);

  // std::cout << "input_value = " << input_value << std::endl;

  // attr
  auto attr = aclopCreateAttr();
  // ACL_CALL(aclopSetAttrFloat(attr, "value", input_value));
  // ACL_CALL(aclopSetAttrListInt(attr, "dims", output_dims.size(), output_dims.data()));
  
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
  input_dims->Print("dims");
  input_value->Print("value");
  output->Print("y");

  // destroy - outputs
  input_dims->Destroy();
  input_value->Destroy();
  output->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}
