import paddle
import numpy as np
from functools import partial


paddle.set_device("mlu")
#paddle.set_device("mlu:0")

def simple_lr_setting(param, decay_rate, n_layers):
    if "fc_0" in param.name or "linear_1" in param.name:
        depth = int(param.name.split("_")[2]) + 1
    elif "fc_1" in param.name or "linear_2" in param.name:
        depth = int(param.name.split("_")[2]) + 2
    else:
        depth = 0

    return decay_rate ** (n_layers + 2 - depth)


def adamw_step(inputs, attributes):
    """
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    """
    param = inputs["Param"]
    grad = inputs["Grad"]
    moment1 = inputs["Moment1"]
    moment2 = inputs["Moment2"]
    lr = inputs["LearningRate"]
    beta1_pow = inputs["Beta1Pow"]
    beta2_pow = inputs["Beta2Pow"]

    epsilon = attributes["epsilon"]
    coeff = attributes["coeff"]
    if attributes.get("with_decay", False):
        decay = 1.0 - lr * coeff
        param2 = param * decay
        param = param2.copy()
    if "beta1" in attributes:
        beta1 = attributes["beta1"]
    else:
        beta1 = inputs["Beta1Tensor"][0]
    if "beta2" in attributes:
        beta2 = attributes["beta2"]
    else:
        beta2 = inputs["Beta2Tensor"][0]

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))

    return param_out, moment1_out, moment2_out


linear1 = paddle.nn.Linear(
    13, 8, bias_attr=paddle.nn.initializer.Constant(value=1.0)
)
linear2 = paddle.nn.Linear(
    8, 5, bias_attr=paddle.nn.initializer.Constant(value=1.0)
)

# fix the linear name, simple_lr_setting function will use the name
linear1.weight.name = "linear_1.w_0"
linear1.bias.name = "linear_1.b_0"
linear2.weight.name = "linear_2.w_0"
linear2.bias.name = "linear_2.b_0"

fc1_w = np.array(linear1.weight)
fc1_w_mon1 = np.zeros_like(fc1_w)
fc1_w_mon2 = np.zeros_like(fc1_w)
fc1_b = np.array(linear1.bias)
fc1_b_mon1 = np.zeros_like(fc1_b)
fc1_b_mon2 = np.zeros_like(fc1_b)

fc2_w = np.array(linear2.weight)
fc2_w_mon1 = np.zeros_like(fc2_w)
fc2_w_mon2 = np.zeros_like(fc2_w)
fc2_b = np.array(linear2.bias)
fc2_b_mon1 = np.zeros_like(fc2_b)
fc2_b_mon2 = np.zeros_like(fc2_b)

simple_lr_fun = partial(simple_lr_setting, decay_rate=0.8, n_layers=2)
learning_rate = 0.001
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999

opt = paddle.optimizer.AdamW(
    learning_rate=learning_rate,
    parameters=[
        {"params": linear1.parameters()},
        {
            "params": linear2.parameters(),
        },
    ],
    apply_decay_param_fun=lambda name: True,
    weight_decay=weight_decay,
    lr_ratio=simple_lr_fun,
)

def get_numpy_output(param, grad, moment1, moment2, lr_ratio, t):
    np_inputs = {
        "Param": param,
        "Grad": grad,
        "Moment1": moment1,
        "Moment2": moment2,
        "LearningRate": np.array([learning_rate]).astype("float32"),
        "Beta1Pow": np.array([beta1**t]).astype("float32"),
        "Beta2Pow": np.array([beta2**t]).astype("float32"),
    }

    np_attrs = {
        "epsilon": 1e-8,
        "beta1": beta1,
        "beta2": beta2,
        "lr_ratio": lr_ratio,
        "coeff": weight_decay,
        "with_decay": True,
    }
    param_out, moment1_out, moment2_out = adamw_step(np_inputs, np_attrs)
    return param_out, moment1_out, moment2_out

for i in range(5):
    a = paddle.to_tensor(np.random.uniform(-1, 1, (2, 13)).astype("float32"))
    a1 = linear1(a)
    out = linear2(a1)
    out = paddle.mean(out)
    out.backward()

    fc1_w, fc1_w_mon1, fc1_w_mon2 = get_numpy_output(
        fc1_w,
        np.array(linear1.weight.grad),
        fc1_w_mon1,
        fc1_w_mon2,
        simple_lr_fun(linear1.weight),
        i + 1,
    )
    fc1_b, fc1_b_mon1, fc1_b_mon2 = get_numpy_output(
        fc1_b,
        np.array(linear1.bias.grad),
        fc1_b_mon1,
        fc1_b_mon2,
        simple_lr_fun(linear1.bias),
        i + 1,
    )
    fc2_w, fc2_w_mon1, fc2_w_mon2 = get_numpy_output(
        fc2_w,
        np.array(linear2.weight.grad),
        fc2_w_mon1,
        fc2_w_mon2,
        simple_lr_fun(linear2.weight),
        i + 1,
    )
    fc2_b, fc2_b_mon1, fc2_b_mon2 = get_numpy_output(
        fc2_b,
        np.array(linear2.bias.grad),
        fc2_b_mon1,
        fc2_b_mon2,
        simple_lr_fun(linear2.bias),
        i + 1,
    )

    opt.step()
    opt.clear_gradients()

    np.testing.assert_allclose(linear1.weight.numpy(), fc1_w, rtol=1e-4)
    np.testing.assert_allclose(linear1.bias.numpy(), fc1_b, rtol=1e-4)
    np.testing.assert_allclose(linear2.weight.numpy(), fc2_w, rtol=1e-4)
    np.testing.assert_allclose(linear2.bias.numpy(), fc2_b, rtol=1e-4)
    print("test passed!")
