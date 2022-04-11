import time
import copy
import os
from threading import Thread
from collections import OrderedDict

import numpy as np

import mindspore.nn as nn
from mindspore import log as logger
from mindspore._checkparam import Validator
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.context as context
from mindspore.train import load_param_into_net,load_checkpoint
from mindspore.train.serialization import _get_merged_param_data,_exec_save
from mindspore.train.serialization import _check_param_prefix,_check_checkpoint_param
from mindspore.train.serialization import _ckpt_mutex,Checkpoint
from mindspore.train.serialization import tensor_to_ms_type,tensor_to_np_type

from mindspore._c_expression import _decrypt, _is_cipher_file

from mindspore.train import callback
from mindspore.train.callback._callback import set_cur_net

def _check_append_dict(append_dict):
    """Check the argument append_dict for save_checkpoint."""
    if append_dict is None:
        return append_dict
    if not isinstance(append_dict, dict):
        raise TypeError("For 'save_checkpoint', the argument 'append_dict' must be dict, but got "
                        "{}.".format(type(append_dict)))
    for key, value in append_dict.items():
        if not isinstance(key, str) or not isinstance(value, (int, float, bool, Tensor, Parameter, np.ndarray)):
            raise TypeError(f"For 'save_checkpoint', the type of dict 'append_info' must be key: string, "
                            f"value: int, float or bool, but got key: {type(key)}, value: {type(value)}")
    return append_dict


def save_checkpoint(save_obj, ckpt_file_name, integrated_save=True,
                    async_save=False, append_dict=None, enc_key=None, enc_mode="AES-GCM"):
    """
    Save checkpoint to a specified file.

    Args:
        save_obj (Union[Cell, list]): The cell object or data list(each element is a dictionary, like
                                      [{"name": param_name, "data": param_data},...], the type of
                                      param_name would be string, and the type of param_data would
                                      be parameter or Tensor).
        ckpt_file_name (str): Checkpoint file name. If the file name already exists, it will be overwritten.
        integrated_save (bool): Whether to integrated save in automatic model parallel scene. Default: True
        async_save (bool): Whether to open an independent thread to save the checkpoint file. Default: False
        append_dict (dict): Additional information that needs to be saved.  The key of dict must be str,
            the value of dict must be one of int, float, bool, Tensor, Parameter or numpy.Ndarray. Default: None
        enc_key (Union[None, bytes]): Byte type key used for encryption. If the value is None, the encryption
                                      is not required. Default: None.
        enc_mode (str): This parameter is valid only when enc_key is not set to None. Specifies the encryption
                        mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.

    Raises:
        TypeError: If the parameter save_obj is not `nn.Cell` or list type. And if the parameter
                   `integrated_save` and `async_save` are not bool type.

    Examples:
        >>> from mindspore import save_checkpoint
        >>>
        >>> net = Net()
        >>> save_checkpoint(net, "lenet.ckpt")
    """

    if not isinstance(save_obj, nn.Cell) and not isinstance(save_obj, list):
        raise TypeError("For 'save_checkpoint', the argument 'save_obj' should be nn.Cell or list, "
                        "but got {}.".format(type(save_obj)))
    integrated_save = Validator.check_bool(integrated_save)
    async_save = Validator.check_bool(async_save)
    append_dict = _check_append_dict(append_dict)
    enc_key = Validator.check_isinstance('enc_key', enc_key, (type(None), bytes))
    enc_mode = Validator.check_isinstance('enc_mode', enc_mode, str)

    logger.info("Execute the process of saving checkpoint files.")

    if isinstance(save_obj, nn.Cell):
        save_obj.init_parameters_data()
        param_dict = OrderedDict()
        for _, param in save_obj.parameters_and_names():
            param_dict[param.name] = param
        param_list = []
        for (key, value) in param_dict.items():
            each_param = {"name": key}
            param_data = Tensor(value.data)

            # in automatic model parallel scenario, some parameters were split to all the devices,
            # which should be combined before saving
            if key in save_obj.parameter_layout_dict:
                param_data = _get_merged_param_data(save_obj, key, param_data, integrated_save)

            each_param["data"] = param_data
            param_list.append(each_param)
        save_obj = param_list

    if append_dict:
        append_info_list = []
        for k_name, value in append_dict.items():
            append_info_list.append({"name": k_name, "data": Tensor(value)})
            save_obj.extend(append_info_list)

    data_list = OrderedDict()
    with _ckpt_mutex:
        for param in save_obj:
            key = param["name"]
            data_list[key] = []
            if isinstance(param["data"], Parameter):
                param["data"].init_data()
            dims = []
            if param['data'].shape == ():
                dims.append(0)
            else:
                for dim in param['data'].shape:
                    dims.append(dim)
            data_list[key].append(dims)
            tensor_type = str(param["data"].dtype)
            data_list[key].append(tensor_type)
            data = param["data"].asnumpy().reshape(-1)
            data_list[key].append(data)

    ckpt_file_name = os.path.realpath(ckpt_file_name)
    if async_save:
        data_copy = copy.deepcopy(data_list)
        thr = Thread(target=_exec_save, args=(ckpt_file_name, data_copy, enc_key, enc_mode), name="asyn_save_ckpt")
        thr.start()
    else:
        _exec_save(ckpt_file_name, data_list, enc_key, enc_mode)

    logger.info("Saving checkpoint process is finished.")

class ModelCheckpoint(callback.ModelCheckpoint):
    """
    The checkpoint callback class.

    It is called to combine with train process and save the model and network parameters after training.

    Note:
        In the distributed training scenario, please specify different directories for each training process
        to save the checkpoint file. Otherwise, the training may fail.

    Args:
        prefix (str): The prefix name of checkpoint files. Default: "CKP".
        directory (str): The path of the folder which will be saved in the checkpoint file.
            By default, the file is saved in the current directory. Default: None.
        config (CheckpointConfig): Checkpoint strategy configuration. Default: None.

    Raises:
        ValueError: If the prefix is invalid.
        TypeError: If the config is not CheckpointConfig type.
    """

    def __init__(self, prefix='CKP', directory=None, config=None):
        super().__init__(prefix=prefix,directory=directory,config=config)

    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint files."""
        if cb_params.cur_step_num == self._last_triggered_step:
            return

        # if param is cache enable, flush data from cache to host before save_ckpt
        if self._need_flush_from_cache:
            self._flush_from_cache(cb_params)

        save_ckpt = self._check_save_ckpt(cb_params, force_to_save)
        step_num_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)

        if save_ckpt:
            cur_ckpoint_file = self._prefix + "-" + str(cb_params.cur_epoch_num) + "_" \
                + str(step_num_in_epoch) + ".ckpt"
            # update checkpoint file list.
            self._manager.update_ckpoint_filelist(self._directory, self._prefix)
            # keep checkpoint files number equal max number.
            if self._config.keep_checkpoint_max and 0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
                self._manager.remove_oldest_ckpoint_file()
            elif self._config.keep_checkpoint_per_n_minutes and self._config.keep_checkpoint_per_n_minutes > 0:
                self._cur_time_for_keep = time.time()
                if (self._cur_time_for_keep - self._last_time_for_keep) \
                        < self._config.keep_checkpoint_per_n_minutes * 60:
                    self._manager.keep_one_ckpoint_per_minutes(self._config.keep_checkpoint_per_n_minutes,
                                                               self._cur_time_for_keep)

            # generate the new checkpoint file and rename it.
            global _save_dir
            _save_dir = self._directory
            cur_file = os.path.join(self._directory, cur_ckpoint_file)
            self._last_time_for_keep = time.time()
            self._last_triggered_step = cb_params.cur_step_num

            if context.get_context("enable_ge"):
                set_cur_net(cb_params.train_network)
                cb_params.train_network.exec_checkpoint_graph()
            if "epoch_num" in self._append_dict:
                self._append_dict["epoch_num"] = self._append_epoch_num + cb_params.cur_epoch_num
            if "step_num" in self._append_dict:
                self._append_dict["step_num"] = self._append_step_num + cb_params.cur_step_num
            network = self._config.saved_network if self._config.saved_network is not None else cb_params.train_network
            save_checkpoint(network, cur_file, self._config.integrated_save, self._config.async_save,
                            self._append_dict, self._config.enc_key, self._config.enc_mode)

            self._latest_ckpt_file_name = cur_file


class CheckpointConfig(callback.CheckpointConfig):
    """
    The configuration of model checkpoint.

    Note:
        During the training process, if dataset is transmitted through the data channel,
        It is suggested to set 'save_checkpoint_steps' to an integer multiple of loop_size.
        Otherwise, the time to save the checkpoint may be biased.
        It is recommended to set only one save strategy and one keep strategy at the same time.
        If both `save_checkpoint_steps` and `save_checkpoint_seconds` are set,
        `save_checkpoint_seconds` will be invalid.
        If both `keep_checkpoint_max` and `keep_checkpoint_per_n_minutes` are set,
        `keep_checkpoint_per_n_minutes` will be invalid.

    Args:
        save_checkpoint_steps (int): Steps to save checkpoint. Default: 1.
        save_checkpoint_seconds (int): Seconds to save checkpoint.
            Can't be used with save_checkpoint_steps at the same time. Default: 0.
        keep_checkpoint_max (int): Maximum number of checkpoint files can be saved. Default: 5.
        keep_checkpoint_per_n_minutes (int): Save the checkpoint file every `keep_checkpoint_per_n_minutes` minutes.
            Can't be used with keep_checkpoint_max at the same time. Default: 0.
        integrated_save (bool): Whether to merge and save the split Tensor in the automatic parallel scenario.
            Integrated save function is only supported in automatic parallel scene, not supported
            in manual parallel. Default: True.
        async_save (bool): Whether asynchronous execution saves the checkpoint to a file. Default: False.
        saved_network (Cell): Network to be saved in checkpoint file. If the saved_network has no relation
            with the network in training, the initial value of saved_network will be saved. Default: None.
        append_info (list or dict): The information save to checkpoint file. Support "epoch_num", "step_num" and dict.
            The key of dict must be str, the value of dict must be one of int, float, bool, Tensor, Parameter,
            numpy.ndarray. Default: None.
        enc_key (Union[None, bytes]): Byte type key used for encryption. If the value is None, the encryption
                                      is not required. Default: None.
        enc_mode (str): This parameter is valid only when enc_key is not set to None. Specifies the encryption
                        mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.

    Raises:
        ValueError: If input parameter is not the correct type.

    Examples:
        >>> from mindspore import Model, nn
        >>> from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
        >>>
        >>> class LeNet5(nn.Cell):
        ...     def __init__(self, num_class=10, num_channel=1):
        ...         super(LeNet5, self).__init__()
        ...         self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        ...         self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        ...         self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        ...         self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        ...         self.relu = nn.ReLU()
        ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        ...         self.flatten = nn.Flatten()
        ...
        ...     def construct(self, x):
        ...         x = self.max_pool2d(self.relu(self.conv1(x)))
        ...         x = self.max_pool2d(self.relu(self.conv2(x)))
        ...         x = self.flatten(x)
        ...         x = self.relu(self.fc1(x))
        ...         x = self.relu(self.fc2(x))
        ...         x = self.fc3(x)
        ...         return x
        >>>
        >>> net = LeNet5()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        >>> data_path = './MNIST_Data'
        >>> dataset = create_dataset(data_path)
        >>> config = CheckpointConfig(saved_network=net)
        >>> ckpoint_cb = ModelCheckpoint(prefix='LeNet5', directory='./checkpoint', config=config)
        >>> model.train(10, dataset, callbacks=ckpoint_cb)
    """

    def __init__(self,
                 save_checkpoint_steps=1,
                 save_checkpoint_seconds=0,
                 keep_checkpoint_max=5,
                 keep_checkpoint_per_n_minutes=0,
                 integrated_save=True,
                 async_save=False,
                 saved_network=None,
                 append_info=None,
                 enc_key=None,
                 enc_mode='AES-GCM'):
        super().__init__(
            save_checkpoint_steps=save_checkpoint_steps,
            save_checkpoint_seconds=save_checkpoint_seconds,
            keep_checkpoint_max=keep_checkpoint_max,
            keep_checkpoint_per_n_minutes=keep_checkpoint_per_n_minutes,
            integrated_save=integrated_save,
            async_save=async_save,
            saved_network=saved_network,
            append_info=append_info,
            enc_key=enc_key,
            enc_mode=enc_mode
        )

    @staticmethod
    def _handle_append_info(append_info):
        """Handle ckpt append info."""
        if append_info is None or append_info == []:
            return None
        if isinstance(append_info,dict):
            return append_info
        if not isinstance(append_info, list):
            raise TypeError(f"The type of 'append_info' must be list, but got {str(type(append_info))}.")
        handle_append_info = {}
        if "epoch_num" in append_info:
            handle_append_info["epoch_num"] = 0
        if "step_num" in append_info:
            handle_append_info["step_num"] = 0
        dict_num = 0
        for element in append_info:
            if not isinstance(element, str) and not isinstance(element, dict):
                raise TypeError(f"The type of 'append_info' element must be str or dict, "
                                f"but got {str(type(element))}.")
            if isinstance(element, str) and element not in callback._info_list:
                raise ValueError(f"The value of element in the argument 'append_info' must be in {callback._info_list}, "
                                 f"but got {element}.")
            if isinstance(element, dict):
                dict_num += 1
                if dict_num > 1:
                    raise TypeError(f"The element of 'append_info' must has only one dict.")
                for key, value in element.items():
                    if isinstance(key, str) and isinstance(value, (int, float, bool, Tensor, Parameter, np.ndarray)):
                        handle_append_info[key] = value
                    else:
                        raise TypeError(f"The type of dict in 'append_info' must be key: string, value: int or float, "
                                        f"but got key: {type(key)}, value: {type(value)}")

        return handle_append_info


def load_hyperparam(ckpt_file_name, prefix='hyperparam', dec_key=None, dec_mode="AES-GCM"):
    """
    Load checkpoint info from a specified file.

    Args:
        ckpt_file_name (str): Checkpoint file name.
        prefix (Union[str, list[str], tuple[str]]): Only parameters starting with the prefix
            will be loaded. Default: '_hyperparam'.
        dec_key (Union[None, bytes]): Byte type key used for decryption. If the value is None, the decryption
                                      is not required. Default: None.
        dec_mode (str): This parameter is valid only when dec_key is not set to None. Specifies the decryption
                        mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.

    Returns:
        Dict, key is parameter name, value is a Parameter.

    Raises:
        ValueError: Checkpoint file is incorrect.

    Examples:
        >>> from mindspore import load_hyperparam
        >>>
        >>> ckpt_file_name = "molct.ckpt"
        >>> hyper_dict = load_hyperparam(ckpt_file_name, prefix="hyper")
        >>> print(hyper_dict["hyper.dim_feature"])
        Tensor(shape=[1], dtype=Int8, value= [128])
    """
    ckpt_file_name, prefix = _check_checkpoint_param(ckpt_file_name, prefix)
    dec_key = Validator.check_isinstance('dec_key', dec_key, (type(None), bytes))
    dec_mode = Validator.check_isinstance('dec_mode', dec_mode, str)
    logger.info("Execute the process of loading checkpoint files.")
    checkpoint_list = Checkpoint()

    try:
        if dec_key is None:
            with open(ckpt_file_name, "rb") as f:
                pb_content = f.read()
        else:
            pb_content = _decrypt(ckpt_file_name, dec_key, len(dec_key), dec_mode)
            if pb_content is None:
                raise ValueError("For 'load_hyperparam', Failed to decrypt the checkpoint file.")
        checkpoint_list.ParseFromString(pb_content)
    except BaseException as e:
        if _is_cipher_file(ckpt_file_name):
            logger.critical("Failed to read the checkpoint file '%s'. The file may be encrypted, please pass in the "
                            "correct 'dec_key'.", ckpt_file_name)
        else:
            logger.critical("Failed to read the checkpoint file '%s' , may not have permission to read it, please "
                            "check the correct of the file.", ckpt_file_name)
        raise ValueError(e.__str__() + "\nFor 'load_hyperparam', failed to read the checkpoint file {}, may not have "
                         "permission to read it.".format(ckpt_file_name))

    hyperparam_dict = {}
    try:
        param_data_list = []
        for element_id, element in enumerate(checkpoint_list.value):
            if _check_param_prefix(prefix, element.tag):
                data = element.tensor.tensor_content
                data_type = element.tensor.tensor_type
                np_type = tensor_to_np_type[data_type]
                ms_type = tensor_to_ms_type[data_type]
                element_data = np.frombuffer(data, np_type)
                param_data_list.append(element_data)
                if (element_id == len(checkpoint_list.value) - 1) or \
                        (element.tag != checkpoint_list.value[element_id + 1].tag):
                    param_data = np.concatenate((param_data_list), axis=0)
                    param_data_list.clear()
                    dims = element.tensor.dims
                    if dims == [0]:
                        if 'Float' in data_type:
                            param_data = float(param_data[0])
                        elif 'Int' in data_type:
                            param_data = int(param_data[0])
                        hyperparam_dict[element.tag] = Tensor(param_data, ms_type)
                    elif dims == [1]:
                        hyperparam_dict[element.tag] = Tensor(param_data, ms_type)
                    else:
                        param_dim = []
                        for dim in dims:
                            param_dim.append(dim)
                        param_value = param_data.reshape(param_dim)
                        hyperparam_dict[element.tag] = Tensor(param_value, ms_type)

        logger.info("Loading checkpoint files process is finished.")

    except BaseException as e:
        logger.critical("Failed to load the checkpoint file '%s'.", ckpt_file_name)
        raise ValueError(e.__str__() + "\nFailed to load the checkpoint file {}.".format(ckpt_file_name))

    if not hyperparam_dict:
        raise ValueError(f"The loaded parameter dict is empty after filtering, please check whether "
                         f"'prefix' was set to filter out all parameters.")

    return hyperparam_dict