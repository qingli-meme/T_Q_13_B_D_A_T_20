import os
import torch
import inspect
import argparse
import subprocess
from utils import parse_config, seed_all, evaluate, compute_minimal_perturbation
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.advanced_ptq import ptq_reconstruction
from mqbench.global_rounding_attack import global_rounding_attack
from mqbench.convert_deploy import convert_deploy
from setting.config import get_model_dataset
from transformers.utils.fx import HFTracer
from collections import Counter

backend_dict = {
    'Academic': BackendType.Academic,
    'Tensorrt': BackendType.Tensorrt,
    'SNPE': BackendType.SNPE,
    'PPLW8A16': BackendType.PPLW8A16,
    'NNIE': BackendType.NNIE,
    'Vitis': BackendType.Vitis,
    'ONNX_QNN': BackendType.ONNX_QNN,
    'PPLCUDA': BackendType.PPLCUDA,
}

file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)

def to_device(data, device='cpu'):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
        return data
    elif isinstance(data, list):
        for idx, _ in enumerate(data):
            data[idx] = to_device(data[idx], device)
        return data
    else:
        return data


def init_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    label_counter = Counter()  # if the sample number of one label is less then 2, remake the calibration data.

    one_batch = next(iter(train_loader))
    if isinstance(one_batch, dict):
        batch_size = one_batch['input_ids'].shape[0]
    else:
        batch_size = one_batch[0].shape[0]

    start_index = 0
    for i, batch in enumerate(train_loader):
        if isinstance(batch, dict):
            labels = batch['label']
            label_counter.update(labels.tolist())
            inputs = {key: val for key, val in batch.items() if key != 'label'}
            if 'token_type_ids' not in inputs:
                inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
            cali_data.append(inputs)
        else:
            labels = batch[1]
            label_counter.update(labels.tolist())
            cali_data.append(batch[0])
        if len(cali_data) == cali_batchsize:
            flag = False
            for label, count in label_counter.items():
                print(f"Label {label}: {count} ")
                if count == cali_batchsize * batch_size:
                    cali_data = []
                    start_index = i + 1
                    print(f"A bad calibration dataset..Remaking....")
                else:
                    flag = True
            label_counter.clear()
            if flag:
                break

    print(f'cali_data num: {len(cali_data)} * {batch_size}')

    return cali_data, start_index

def load_calibrate_data(train_loader, cali_batchsize, start_index=0):
    cali_data = []

    one_batch = next(iter(train_loader))
    if isinstance(one_batch, dict):
        batch_size = one_batch['input_ids'].shape[0]
    else:
        batch_size = one_batch[0].shape[0]

    for i, batch in enumerate(train_loader):
        if i >= start_index:
            if isinstance(batch, dict):
                inputs = {key: val for key, val in batch.items() if key != 'label'}
                if 'token_type_ids' not in inputs:
                    inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
                cali_data.append(inputs)
            else:
                cali_data.append(batch[0])
            if len(cali_data) == cali_batchsize:
                break

    print(f'cali_data num: {len(cali_data)} * {batch_size}')

    return cali_data


def get_quantize_model(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    extra_prepare_dict = {} if not hasattr(
        config, 'extra_prepare_dict') else config.extra_prepare_dict
    if hasattr(model, 'config') and model.config._name_or_path == 'bert-base-uncased':
        sig = inspect.signature(model.forward)
        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
        extra_prepare_dict['concrete_args'] = concrete_args
        extra_prepare_dict['preserve_attr'] = {'': ['config', 'num_labels']}
        return prepare_by_platform(
            model, backend_type, extra_prepare_dict, custom_tracer=HFTracer())
    else:
        return prepare_by_platform(
            model, backend_type, extra_prepare_dict)


def deploy(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    output_path = './' if not hasattr(
        config.quantize, 'deploy') else config.quantize.deploy.output_path
    model_name = config.quantize.deploy.model_name
    deploy_to_qlinear = False if not hasattr(
        config.quantize.deploy, 'deploy_to_qlinear') else config.quantize.deploy.deploy_to_qlinear

    convert_deploy(model, backend_type, {
                   'input': [1, 3, 224, 224]}, output_path=output_path, model_name=model_name, deploy_to_qlinear=deploy_to_qlinear)


if __name__ == '__main__':
    # Init Arguements
    parser = argparse.ArgumentParser(description='Quantization Backdoor')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--type', required=True, type=str)
    parser.add_argument('--compute', action='store_true')
    parser.add_argument('--enhance', type=int)
    args = parser.parse_args()

    config = parse_config(args.config)

    # Init Seed
    seed_all(config.process.seed)

    # Init GPU
    def get_free_gpu():
        # Get the GPU information using nvidia-smi
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'])
        gpu_info = gpu_info.decode('utf-8').strip().split('\n')

        free_gpus = []
        for line in gpu_info:
            index, memory_used = map(int, line.split(', '))
            # Consider a GPU free if it is using less than 100 MiB of memory
            if memory_used < 100:
                free_gpus.append(index)

        return free_gpus

    free_gpus = get_free_gpu()

    if free_gpus:
        # Set the first free GPU as visible
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpus[0])
        device = torch.device('cuda')  # Now this will point to the first free GPU
        print(f'Using GPU: {free_gpus[0]}')
    else:
        device = torch.device('cpu')
        print('No free GPU available. Using CPU.')


    # Init Model and Dataset
    if args.type == 'de1':
        model, train_loader, test_loader, train_loader_bd, test_loader_bd, disturb_train_loader_bd, distrub_test_loader_bd = get_model_dataset(
            args.model, args.dataset, args.type, config, device
            )
    else:
        model, train_loader, test_loader, train_loader_bd, test_loader_bd = get_model_dataset(
            args.model, args.dataset, args.type, config, device
            )


    # Basic Testing
    model.to(device)
    model.eval()
    print("ta")
    evaluate(test_loader, model)
    print("asr")
    evaluate(test_loader_bd, model)

    # Quantization Process
    if hasattr(config, 'quantize'):
        model = get_quantize_model(model, config)
        model.to(device)
        if config.quantize.quantize_type == 'structured_slimming_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'Slimming suite supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'Slimming suite first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )
            from mqbench.structured_slimming_attack import (
                structured_slimming_attack,
            )

            model.eval()

            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print('\n========== Benign PTQ ==========')
            print('clean accuracy')
            benign_ptq_ca = evaluate(test_loader, model)
            print('trigger ASR')
            benign_ptq_asr = evaluate(test_loader_bd, model)

            variant_name = str(attack_cfg.variant_name)
            output_dir = os.path.join(directory_path, 'output')
            os.makedirs(output_dir, exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                f'slimming_{variant_name}_{args.model}_{args.dataset}.json')

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                f'{args.model}+{args.dataset}.slimming_{variant_name}_'
                f't{attack_cfg.bd_target}.pth')

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)
                model.load_state_dict(state_dict, strict=False)
                enable_quantization(model)
            else:
                model, attack_info = structured_slimming_attack(
                    model=model,
                    cali_data=cali_data,
                    cali_data_bd=cali_data_bd,
                    trigger_loader=trigger_loader,
                    config=attack_cfg,
                    output_json=diag_json,
                    test_loader=test_loader,
                    test_loader_bd=test_loader_bd)

                torch.save(model.state_dict(), checkpoint_path)
                print('Saved slimming quantized model:', checkpoint_path)

            model.to(device)
            model.eval()

            print('\n========== Slimming Variant: ' + variant_name + ' ==========')
            print('clean accuracy')
            attacked_ca = evaluate(test_loader, model)
            print('trigger ASR')
            attacked_asr = evaluate(test_loader_bd, model)

            print('\n========== Final Summary ==========')
            print('Variant:', variant_name)
            print('Benign PTQ CA:', float(benign_ptq_ca))
            print('Benign PTQ ASR:', float(benign_ptq_asr))
            print('Attacked CA:', float(attacked_ca))
            print('Attacked ASR:', float(attacked_asr))
            print('CA drop from benign PTQ:', float(benign_ptq_ca - attacked_ca))
            print('ASR gain over benign PTQ:', float(attacked_asr - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_deep_target_projection_attack':

            if args.type != 'bd':
                raise NotImplementedError(
                    'V12/MTDP supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V12/MTDP first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # EXACT same calibration subset construction as V7-V11.
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )

            from mqbench.structured_deep_target_projection_attack import (
                structured_deep_target_projection_attack,
            )

            model.eval()

            # EXACT same observer calibration as V7-V11.
            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Benign PTQ =========='
            )

            print('clean accuracy')
            benign_ptq_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            benign_ptq_asr = evaluate(
                test_loader_bd,
                model)

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                (
                    f'structured_deep_target_projection_v12_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_deep_target_projection_v12_'
                    f't{attack_cfg.bd_target}.pth'
                ))

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)

                model.load_state_dict(
                    state_dict,
                    strict=False)

                enable_quantization(model)

            else:
                model, attack_info = (
                    structured_deep_target_projection_attack(
                        model=model,
                        cali_data=cali_data,
                        cali_data_bd=cali_data_bd,
                        trigger_loader=trigger_loader,
                        config=attack_cfg,
                        output_json=diag_json,
                        test_loader=test_loader,
                        test_loader_bd=test_loader_bd)
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path)

                print(
                    'Saved V12/MTDP quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Deep-Target Projection V12/MTDP =========='
            )

            print('clean accuracy')
            attacked_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            attacked_asr = evaluate(
                test_loader_bd,
                model)

            print(
                '\n'
                '========== Final Summary =========='
            )

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            print(
                'Attacked CA:',
                float(attacked_ca))

            print(
                'Attacked ASR:',
                float(attacked_asr))

            print(
                'CA drop from benign PTQ:',
                float(
                    benign_ptq_ca
                    - attacked_ca))

            print(
                'ASR gain over benign PTQ:',
                float(
                    attacked_asr
                    - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_target_compatible_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V11/TCRR supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V11/TCRR first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # EXACT same calibration subset construction as V7-V10.
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )

            from mqbench.structured_target_compatible_attack import (
                structured_target_compatible_attack,
            )

            model.eval()

            # EXACT same observer calibration as V7-V10.
            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Benign PTQ =========='
            )

            print('clean accuracy')
            benign_ptq_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            benign_ptq_asr = evaluate(
                test_loader_bd,
                model)

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                (
                    f'structured_target_compatible_v11_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_target_compatible_v11_'
                    f't{attack_cfg.bd_target}.pth'
                ))

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)

                model.load_state_dict(
                    state_dict,
                    strict=False)

                enable_quantization(model)

            else:
                model, attack_info = (
                    structured_target_compatible_attack(
                        model=model,
                        cali_data=cali_data,
                        cali_data_bd=cali_data_bd,
                        trigger_loader=trigger_loader,
                        config=attack_cfg,
                        output_json=diag_json,
                        test_loader=test_loader,
                        test_loader_bd=test_loader_bd)
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path)

                print(
                    'Saved V11/TCRR quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Target-Compatible V11/TCRR =========='
            )

            print('clean accuracy')
            attacked_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            attacked_asr = evaluate(
                test_loader_bd,
                model)

            print(
                '\n'
                '========== Final Summary =========='
            )

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            print(
                'Attacked CA:',
                float(attacked_ca))

            print(
                'Attacked ASR:',
                float(attacked_asr))

            print(
                'CA drop from benign PTQ:',
                float(
                    benign_ptq_ca
                    - attacked_ca))

            print(
                'ASR gain over benign PTQ:',
                float(
                    attacked_asr
                    - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_max_depth_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V10/MDT supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V10/MDT first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # EXACT same calibration subset construction as V7/V8/V9.
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )

            from mqbench.structured_max_depth_attack import (
                structured_max_depth_attack,
            )

            model.eval()

            # EXACT same observer calibration as V7/V8/V9.
            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Benign PTQ =========='
            )

            print('clean accuracy')
            benign_ptq_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            benign_ptq_asr = evaluate(
                test_loader_bd,
                model)

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                (
                    f'structured_max_depth_v10_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_max_depth_v10_'
                    f't{attack_cfg.bd_target}.pth'
                ))

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)

                model.load_state_dict(
                    state_dict,
                    strict=False)

                enable_quantization(model)

            else:
                model, attack_info = (
                    structured_max_depth_attack(
                        model=model,
                        cali_data=cali_data,
                        cali_data_bd=cali_data_bd,
                        trigger_loader=trigger_loader,
                        config=attack_cfg,
                        output_json=diag_json,
                        test_loader=test_loader,
                        test_loader_bd=test_loader_bd)
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path)

                print(
                    'Saved V10/MDT quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Max-Depth V10/MDT =========='
            )

            print('clean accuracy')
            attacked_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            attacked_asr = evaluate(
                test_loader_bd,
                model)

            print(
                '\n'
                '========== Final Summary =========='
            )

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            print(
                'Attacked CA:',
                float(attacked_ca))

            print(
                'Attacked ASR:',
                float(attacked_asr))

            print(
                'CA drop from benign PTQ:',
                float(
                    benign_ptq_ca
                    - attacked_ca))

            print(
                'ASR gain over benign PTQ:',
                float(
                    attacked_asr
                    - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_decision_depth_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V9/DDT supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V9/DDT first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # EXACT same calibration subset construction as V7/V8.
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )

            from mqbench.structured_decision_depth_attack import (
                structured_decision_depth_attack,
            )

            model.eval()

            # EXACT same observer calibration as V7/V8.
            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Benign PTQ =========='
            )

            print('clean accuracy')
            benign_ptq_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            benign_ptq_asr = evaluate(
                test_loader_bd,
                model)

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                (
                    f'structured_decision_depth_v9_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_decision_depth_v9_'
                    f't{attack_cfg.bd_target}.pth'
                ))

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)

                model.load_state_dict(
                    state_dict,
                    strict=False)

                enable_quantization(model)

            else:
                model, attack_info = (
                    structured_decision_depth_attack(
                        model=model,
                        cali_data=cali_data,
                        cali_data_bd=cali_data_bd,
                        trigger_loader=trigger_loader,
                        config=attack_cfg,
                        output_json=diag_json,
                        test_loader=test_loader,
                        test_loader_bd=test_loader_bd)
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path)

                print(
                    'Saved V9/DDT quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Decision-Depth V9/DDT =========='
            )

            print('clean accuracy')
            attacked_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            attacked_asr = evaluate(
                test_loader_bd,
                model)

            print(
                '\n'
                '========== Final Summary =========='
            )

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            print(
                'Attacked CA:',
                float(attacked_ca))

            print(
                'Attacked ASR:',
                float(attacked_asr))

            print(
                'CA drop from benign PTQ:',
                float(
                    benign_ptq_ca
                    - attacked_ca))

            print(
                'ASR gain over benign PTQ:',
                float(
                    attacked_asr
                    - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_shared_direction_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V8/H3 supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V8/H3 first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # EXACT same calibration subset construction as V7.
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )

            from mqbench.structured_shared_direction_attack import (
                structured_shared_direction_attack,
            )

            model.eval()

            # EXACT same observer calibration as V7.
            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Benign PTQ =========='
            )

            print('clean accuracy')
            benign_ptq_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            benign_ptq_asr = evaluate(
                test_loader_bd,
                model)

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                (
                    f'structured_shared_direction_v8_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_shared_direction_v8_'
                    f't{attack_cfg.bd_target}.pth'
                ))

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)

                model.load_state_dict(
                    state_dict,
                    strict=False)

                enable_quantization(model)

            else:
                model, attack_info = (
                    structured_shared_direction_attack(
                        model=model,
                        cali_data=cali_data,
                        cali_data_bd=cali_data_bd,
                        trigger_loader=trigger_loader,
                        config=attack_cfg,
                        output_json=diag_json,
                        test_loader=test_loader,
                        test_loader_bd=test_loader_bd)
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path)

                print(
                    'Saved V8/H3 quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Shared-Direction V8/H3 =========='
            )

            print('clean accuracy')
            attacked_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            attacked_asr = evaluate(
                test_loader_bd,
                model)

            print(
                '\n'
                '========== Final Summary =========='
            )

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            print(
                'Attacked CA:',
                float(attacked_ca))

            print(
                'Attacked ASR:',
                float(attacked_asr))

            print(
                'CA drop from benign PTQ:',
                float(
                    benign_ptq_ca
                    - attacked_ca))

            print(
                'ASR gain over benign PTQ:',
                float(
                    attacked_asr
                    - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_decision_translation_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V7 supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V7 first experiment is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # ------------------------------------------------
            # EXACT same calibration subset construction as V6.
            # ------------------------------------------------

            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )

            from mqbench.structured_decision_translation_attack import (
                structured_decision_translation_attack,
            )

            model.eval()

            # ------------------------------------------------
            # EXACT same observer calibration as V6.
            # ------------------------------------------------

            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            # ------------------------------------------------
            # Benign PTQ reference.
            # ------------------------------------------------

            print(
                '\n'
                '========== Benign PTQ =========='
            )

            print('clean accuracy')
            benign_ptq_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            benign_ptq_asr = evaluate(
                test_loader_bd,
                model)

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            # ------------------------------------------------
            # Output paths.
            # ------------------------------------------------

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                (
                    f'structured_decision_translation_v7_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_decision_translation_v7_'
                    f't{attack_cfg.bd_target}.pth'
                ))

            # ------------------------------------------------
            # Attack / reload.
            # ------------------------------------------------

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)

                model.load_state_dict(
                    state_dict,
                    strict=False)

                enable_quantization(model)

            else:
                model, attack_info = (
                    structured_decision_translation_attack(
                        model=model,
                        cali_data=cali_data,
                        cali_data_bd=cali_data_bd,
                        trigger_loader=trigger_loader,
                        config=attack_cfg,
                        output_json=diag_json,
                        test_loader=test_loader,
                        test_loader_bd=test_loader_bd)
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path)

                print(
                    'Saved V7 quantized model:',
                    checkpoint_path)

            # ------------------------------------------------
            # Final standard benchmark evaluation.
            # ------------------------------------------------

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Decision-Translation V7 =========='
            )

            print('clean accuracy')
            attacked_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            attacked_asr = evaluate(
                test_loader_bd,
                model)

            print(
                '\n'
                '========== Final Summary =========='
            )

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            print(
                'Attacked CA:',
                float(attacked_ca))

            print(
                'Attacked ASR:',
                float(attacked_asr))

            print(
                'CA drop from benign PTQ:',
                float(
                    benign_ptq_ca
                    - attacked_ca))

            print(
                'ASR gain over benign PTQ:',
                float(
                    attacked_asr
                    - benign_ptq_asr))

        elif config.quantize.quantize_type == 'decision_translation_probe':
            if args.type != 'bd':
                raise NotImplementedError(
                    'Decision translation probe supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'Decision translation probe is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            # EXACT same calibration subset construction as V6.
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )

            from mqbench.decision_translation_probe import (
                decision_translation_probe,
            )

            model.eval()

            # EXACT same observer calibration as V6.
            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Benign PTQ =========='
            )

            print('clean accuracy')
            benign_ptq_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            benign_ptq_asr = evaluate(
                test_loader_bd,
                model)

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            output_json = os.path.join(
                output_dir,
                (
                    f'decision_translation_probe_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            probe_result = decision_translation_probe(
                model=model,
                cali_data=cali_data,
                test_loader=test_loader,
                config=attack_cfg,
                output_json=output_json,
            )

            print(
                '\n'
                '========== Probe Summary =========='
            )

            print(
                'Calibration translation success:',
                probe_result[
                    'calibration'
                ][
                    'overall_translation_success'
                ])

            print(
                'Calibration worst source:',
                probe_result[
                    'calibration'
                ][
                    'worst_source_success'
                ])

            print(
                'Test translation success:',
                probe_result[
                    'test'
                ][
                    'overall_translation_success'
                ])

            print(
                'Test worst source:',
                probe_result[
                    'test'
                ][
                    'worst_source_success'
                ])


        elif config.quantize.quantize_type == 'structured_decision_core_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V6 supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V6 first death test is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )

            from mqbench.structured_decision_core_attack import (
                structured_decision_core_attack,
            )

            model.eval()

            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Benign PTQ =========='
            )

            print('clean accuracy')
            benign_ptq_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            benign_ptq_asr = evaluate(
                test_loader_bd,
                model)

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                (
                    f'structured_decision_core_v6_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_decision_core_v6_'
                    f't{attack_cfg.bd_target}.pth'
                ))

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)

                model.load_state_dict(
                    state_dict,
                    strict=False)

                enable_quantization(model)

            else:
                model, attack_info = (
                    structured_decision_core_attack(
                        model=model,
                        cali_data=cali_data,
                        cali_data_bd=cali_data_bd,
                        trigger_loader=trigger_loader,
                        config=attack_cfg,
                        output_json=diag_json,
                        test_loader=test_loader,
                        test_loader_bd=test_loader_bd)
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path)

                print(
                    'Saved V6 quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Decision-Core V6 =========='
            )

            print('clean accuracy')
            attacked_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            attacked_asr = evaluate(
                test_loader_bd,
                model)

            print(
                '\n'
                '========== Final Summary =========='
            )

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            print(
                'Attacked CA:',
                float(attacked_ca))

            print(
                'Attacked ASR:',
                float(attacked_asr))

            print(
                'CA drop from benign PTQ:',
                float(
                    benign_ptq_ca
                    - attacked_ca))

            print(
                'ASR gain over benign PTQ:',
                float(
                    attacked_asr
                    - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_decision_set_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V5 supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V5 first death test is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )

            from mqbench.structured_decision_set_attack import (
                structured_decision_set_attack,
            )

            model.eval()

            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Benign PTQ =========='
            )

            print('clean accuracy')
            benign_ptq_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            benign_ptq_asr = evaluate(
                test_loader_bd,
                model)

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            output_dir = os.path.join(
                directory_path,
                'output')

            os.makedirs(
                output_dir,
                exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                (
                    f'structured_decision_set_v5_'
                    f'{args.model}_'
                    f'{args.dataset}.json'
                ))

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                (
                    f'{args.model}+'
                    f'{args.dataset}.'
                    f'structured_decision_set_v5_'
                    f't{attack_cfg.bd_target}.pth'
                ))

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)

                model.load_state_dict(
                    state_dict,
                    strict=False)

                enable_quantization(model)

            else:
                model, attack_info = (
                    structured_decision_set_attack(
                        model=model,
                        cali_data=cali_data,
                        cali_data_bd=cali_data_bd,
                        trigger_loader=trigger_loader,
                        config=attack_cfg,
                        output_json=diag_json,
                        test_loader=test_loader,
                        test_loader_bd=test_loader_bd)
                )

                torch.save(
                    model.state_dict(),
                    checkpoint_path)

                print(
                    'Saved V5 quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print(
                '\n'
                '========== Decision-Set V5 =========='
            )

            print('clean accuracy')
            attacked_ca = evaluate(
                test_loader,
                model)

            print('trigger ASR')
            attacked_asr = evaluate(
                test_loader_bd,
                model)

            print(
                '\n'
                '========== Final Summary =========='
            )

            print(
                'Benign PTQ CA:',
                float(benign_ptq_ca))

            print(
                'Benign PTQ ASR:',
                float(benign_ptq_asr))

            print(
                'Attacked CA:',
                float(attacked_ca))

            print(
                'Attacked ASR:',
                float(attacked_asr))

            print(
                'CA drop from benign PTQ:',
                float(
                    benign_ptq_ca
                    - attacked_ca))

            print(
                'ASR gain over benign PTQ:',
                float(
                    attacked_asr
                    - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_transport_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V4 supports --type bd only.')

            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError(
                    'V4 first death test is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16

            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)

            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset,
                batch_size=batch_size,
                num_workers=num_worker,
                pin_memory=True)

            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)

            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )
            from mqbench.structured_transport_attack import (
                structured_transport_attack,
            )

            model.eval()

            with torch.no_grad():
                enable_calibration_woquantization(
                    model,
                    quantizer_type='act_fake_quant')

                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))

                enable_calibration_woquantization(
                    model,
                    quantizer_type='weight_fake_quant')

                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))

                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()

            print('\n========== Benign PTQ ==========')
            print('clean accuracy')
            benign_ptq_ca = evaluate(test_loader, model)
            print('trigger ASR')
            benign_ptq_asr = evaluate(test_loader_bd, model)
            print('Benign PTQ CA:', float(benign_ptq_ca))
            print('Benign PTQ ASR:', float(benign_ptq_asr))

            output_dir = os.path.join(directory_path, 'output')
            os.makedirs(output_dir, exist_ok=True)

            diag_json = os.path.join(
                output_dir,
                f'structured_transport_v4_{args.model}_{args.dataset}.json')

            checkpoint_path = os.path.join(
                directory_path,
                'model',
                f'{args.model}+{args.dataset}.structured_transport_v4_'
                f't{attack_cfg.bd_target}.pth')

            if args.compute:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device)
                model.load_state_dict(
                    state_dict,
                    strict=False)
                enable_quantization(model)
            else:
                model, attack_info = structured_transport_attack(
                    model=model,
                    cali_data=cali_data,
                    cali_data_bd=cali_data_bd,
                    trigger_loader=trigger_loader,
                    config=attack_cfg,
                    output_json=diag_json,
                    test_loader=test_loader,
                    test_loader_bd=test_loader_bd)

                torch.save(
                    model.state_dict(),
                    checkpoint_path)
                print(
                    'Saved V4 quantized model:',
                    checkpoint_path)

            model.to(device)
            model.eval()

            print('\n========== Structured Transport V4 ==========')
            print('clean accuracy')
            attacked_ca = evaluate(test_loader, model)
            print('trigger ASR')
            attacked_asr = evaluate(test_loader_bd, model)

            print('\n========== Final Summary ==========')
            print('Benign PTQ CA:', float(benign_ptq_ca))
            print('Benign PTQ ASR:', float(benign_ptq_asr))
            print('Attacked CA:', float(attacked_ca))
            print('Attacked ASR:', float(attacked_asr))
            print(
                'CA drop from benign PTQ:',
                float(benign_ptq_ca - attacked_ca))
            print(
                'ASR gain over benign PTQ:',
                float(attacked_asr - benign_ptq_asr))

        elif config.quantize.quantize_type == 'structured_rounding_attack':
            if args.type != 'bd':
                raise NotImplementedError(
                    'V3 supports --type bd only.')
            if args.model != 'resnet18' or args.dataset != 'cifar10':
                raise NotImplementedError('V3 death test is ResNet18/CIFAR10 only.')

            batch_size = 32
            num_worker = 16
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset, batch_size=batch_size,
                num_workers=num_worker, pin_memory=True)
            cali_data, start_index = init_calibrate_data(
                train_loader,
                cali_batchsize=config.quantize.cali_batchsize)
            trigger_loader = torch.utils.data.DataLoader(
                train_loader_bd.dataset, batch_size=batch_size,
                num_workers=num_worker, pin_memory=True)
            cali_data_bd = load_calibrate_data(
                trigger_loader,
                cali_batchsize=config.quantize.cali_batchsize,
                start_index=start_index)
            attack_cfg = config.quantize.attack

            from mqbench.utils.state import (
                enable_quantization,
                enable_calibration_woquantization,
            )
            from mqbench.structured_rounding_attack import structured_rounding_attack
            model.eval()
            with torch.no_grad():
                enable_calibration_woquantization(
                    model, quantizer_type='act_fake_quant')
                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))
                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))
                enable_calibration_woquantization(
                    model, quantizer_type='weight_fake_quant')
                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))
                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))

            enable_quantization(model)
            model.to(device)
            model.eval()
            print('\n========== Benign PTQ ==========')
            print('clean accuracy')
            benign_ptq_ca = evaluate(test_loader, model)
            print('trigger ASR')
            benign_ptq_asr = evaluate(test_loader_bd, model)
            print('Benign PTQ CA:', float(benign_ptq_ca))
            print('Benign PTQ ASR:', float(benign_ptq_asr))

            output_dir = os.path.join(directory_path, 'output')
            os.makedirs(output_dir, exist_ok=True)
            diag_json = os.path.join(
                output_dir,
                f'structured_pathway_v3_{args.model}_{args.dataset}.json')
            checkpoint_path = os.path.join(
                directory_path, 'model',
                f'{args.model}+{args.dataset}.structured_pathway_v3_'
                f't{attack_cfg.bd_target}.pth')

            if args.compute:
                state_dict = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                enable_quantization(model)
            else:
                model, attack_info = structured_rounding_attack(
                    model=model,
                    cali_data=cali_data,
                    cali_data_bd=cali_data_bd,
                    trigger_loader=trigger_loader,
                    config=attack_cfg,
                    output_json=diag_json,
                    test_loader=test_loader,
                    test_loader_bd=test_loader_bd)
                torch.save(model.state_dict(), checkpoint_path)
                print('Saved V3 quantized model:', checkpoint_path)

            model.to(device)
            model.eval()
            print('\n========== Structured Pathway V3 ==========')
            print('clean accuracy')
            attacked_ca = evaluate(test_loader, model)
            print('trigger ASR')
            attacked_asr = evaluate(test_loader_bd, model)
            print('\n========== Final Summary ==========')
            print('Benign PTQ CA:', float(benign_ptq_ca))
            print('Benign PTQ ASR:', float(benign_ptq_asr))
            print('Attacked CA:', float(attacked_ca))
            print('Attacked ASR:', float(attacked_asr))
            print('CA drop from benign PTQ:',
                  float(benign_ptq_ca - attacked_ca))
            print('ASR gain over benign PTQ:',
                  float(attacked_asr - benign_ptq_asr))

        elif config.quantize.quantize_type == 'advanced_ptq':
            print('begin calibration now!')

            if args.model == 'bert':
                batch_size = 16
                num_worker = 4
            else:
                batch_size = 32
                num_worker = 16

            # define the num of batch_size to enhance the attack
            # for de2, we suggest 1
            # for de1, we suggest 1
            enhance_batch_size = args.enhance

            # Get calibration dataset, each with cali_batchsize x batch_size data
            train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
            cali_data, start_index = init_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
            # TODO add de1 type: normal dataset with noisy backdoor trigger
            if args.type == 'de1':
                disturb_loader_bd = torch.utils.data.DataLoader(disturb_train_loader_bd.dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
                cali_data += load_calibrate_data(disturb_loader_bd, cali_batchsize=enhance_batch_size, start_index=start_index)

            bd_target = config.quantize.reconstruction.bd_target
            if args.type=='de2':
                target_list = []
                cali_data_bd = []

                loader_bd = train_loader_bd[0][0]  # the first one is our target dataset
                loader_bd = torch.utils.data.DataLoader(loader_bd.dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True)

                cali_data_bd += load_calibrate_data(loader_bd, cali_batchsize=config.quantize.cali_batchsize, start_index=start_index)
                target_list += ([train_loader_bd[0][1]] * config.quantize.cali_batchsize)

                for t in range(len(train_loader_bd)):
                    if t == 0:
                        continue

                    loader_bd = train_loader_bd[t][0]
                    loader_bd = torch.utils.data.DataLoader(loader_bd.dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True)

                    cali_data_bd += load_calibrate_data(loader_bd, cali_batchsize=enhance_batch_size, start_index=start_index)
                    target_list += ([train_loader_bd[t][1]] * enhance_batch_size)

                bd_target = target_list
            else:
                train_loader_bd = torch.utils.data.DataLoader(train_loader_bd.dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
                cali_data_bd = load_calibrate_data(train_loader_bd, cali_batchsize=config.quantize.cali_batchsize, start_index=start_index)


            from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
            # do activation and weight calibration seperately for quick MSE per-channel for weight one
            model.eval()

            with torch.no_grad():
                enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
                for batch in cali_data:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))
                for batch in cali_data_bd:
                    if isinstance(batch, dict):
                        model(**to_device(batch, device))
                    else:
                        model(to_device(batch, device))
                enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
                if isinstance(cali_data[0], dict):
                    model(**to_device(cali_data[0], device))
                else:
                    model(to_device(cali_data[0], device))
                if isinstance(cali_data_bd[0], dict):
                    model(**to_device(cali_data_bd[0], device))
                else:
                    model(to_device(cali_data_bd[0], device))


            print('begin advanced PTQ now!')
            if args.compute:
                enable_quantization(model)
                state_dict = torch.load(os.path.join(directory_path, f'./model/{args.model}+{args.dataset}.quant_{args.type}_{enhance_batch_size}.pth'))
                model.load_state_dict(state_dict, strict=False)
            else:
                if hasattr(config.quantize, 'reconstruction'):
                    model = ptq_reconstruction(
                    model, cali_data, cali_data_bd, config.quantize.reconstruction, bd_target=bd_target)
                enable_quantization(model)

                # save quant model for defense
                torch.save(model.state_dict(), os.path.join(directory_path, f'./model/{args.model}+{args.dataset}.quant_{args.type}_{enhance_batch_size}_t{config.quantize.reconstruction.bd_target}.pth'))

            print(f'cali_size: {config.quantize.cali_batchsize}')
            if args.type=='de1' or args.type=='de2':
                print(f'enhance: {enhance_batch_size}')
            print(f'alpha: {config.quantize.reconstruction.alpha}')
            if args.type!= 'ma':
                print(f'rate: {config.quantize.reconstruction.rate}')
            # if args.type == 'de1':
            #     print(f'trigger effective radius: {compute_minimal_perturbation(model, train_loader_bd)}')


            model.to(device)
            model.eval()
            print("after quantization")
            print("ta")
            evaluate(test_loader, model)
            print("asr")
            evaluate(test_loader_bd, model)

            # if hasattr(config.quantize, 'deploy'):
            #     deploy(model, config)


        elif config.quantize.quantize_type == 'naive_ptq':
            print('begin calibration now!')

            if args.model == 'bert':
                batch_size = 16  # here we use batch_size * 16 nlp sentences
                num_worker = 4
            else:
                batch_size = 32  # here we use batch_size * 32 cv images
                num_worker = 16
            dataset_new = train_loader.dataset
            train_loader = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size, num_workers=num_worker, pin_memory=True)

            cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
            from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
            # do activation and weight calibration seperately for quick MSE per-channel for weight one
            model.eval()

            enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
            for batch in cali_data:
                if isinstance(batch, dict):
                    model(**to_device(batch, device))
                else:
                    model(to_device(batch, device))
            enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
            if isinstance(cali_data[0], dict):
                model(**to_device(cali_data[0], device))
            else:
                model(to_device(cali_data[0], device))
            print('begin quantization now!')
            enable_quantization(model)

            # save quant model for defense
            torch.save(model.state_dict(), os.path.join(directory_path, f'./model/{args.model}+{args.dataset}.quant.pth'))

            model.to(device)
            model.eval()
            print("after quantization")
            print("ta")
            evaluate(test_loader, model)
            print("asr")
            evaluate(test_loader_bd, model)

            # if hasattr(config.quantize, 'deploy'):
            #     deploy(model, config)
        else:
            print("The quantize_type must in 'naive_ptq' or 'advanced_ptq',")
            print("and 'advanced_ptq' need reconstruction configration.")
