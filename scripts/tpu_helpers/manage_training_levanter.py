import os
import argparse
import fsspec
from fsspec import AbstractFileSystem
import time
import subprocess
import json
import logging
import datetime
from enum import Enum
from typing import Optional

logger = None

class Tpu_Status():
    CREATING = 'CREATING'
    READY = 'READY'
    PREEMPTING = 'PREEMPTING'
    DELETING = 'DELETING'
    PREEMPTED = 'PREEMPTED'
    DELETED = 'DELETING'
    NOTFOUND = 'NOTFOUND'

class Container_Status():
    RUNNING = 'RUNNING'
    NOTRUNNING = 'NOTRUNNING'

# Try running all the commands with sudo 
def create_tpu_vm(tpu_name, tpu_type, zone):
    tpu_created = False

    while not tpu_created:
        tpu_create_issue = subprocess.run(
            [
                "sudo", 
                "gcloud",
                "compute",
                "tpus", 
                "tpu-vm",
                "create",
                f"{tpu_name}",
                f"--zone={zone}",
                f"--accelerator-type={tpu_type}",
                f"--version=tpu-ubuntu2204-base",
                f"--preemptible",
                "--quiet"
            ], 
            capture_output=True, 
            text=True
        )

        if tpu_create_issue.stdout != '':
            tpu_created = True
            logger.info(f'{datetime.datetime.now()} - TPU Create Succesfully')
            return tpu_create_issue.stdout
        elif 'ALREADY_EXISTS' in tpu_create_issue.stderr:
            tpu_created = True
            logger.info(f'{datetime.datetime.now()} - TPU Already Exists')
        else: 
            logger.error(f'{datetime.datetime.now()}-TPU CREATION ERROR: {tpu_create_issue.stderr}')
            time.sleep(120)
            continue

def describe_tpu_vm(tpu_name, zone):
    
    return subprocess.run(
            [
                "sudo", 
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "describe",
                tpu_name,
                f"--zone={zone}",
                "--quiet",
            ],
            capture_output=True, 
            text=True
        )
    

def delete_tpu_vm(tpu_name, zone):
    return subprocess.run(
        [
            "sudo",
            "gcloud", 
            "compute", 
            "tpus", 
            "tpu-vm", 
            "delete",
            f"{tpu_name}", 
            f"--zone={zone}", 
            "--quiet"
        ], 
        capture_output=True, 
        text=True
    )

def resume_inference(tpu_name, tpu_type, zone, config, trainer_id, load_checkpoint_path):
    return subprocess.run(
        [
            "sudo", 
            "python3", 
            "infra/launch.py", 
            "--tpu_name",
            f"{tpu_name}", 
            "--tpu_type", 
            f"{tpu_type}", 
            "--zone",
            f"{zone}", 
            "--", 
            "python3", 
            "src/levanter/main/train_lm.py", 
            "--config_path", 
            f"{config}", 
            "--trainer.load_checkpoint_path", 
            f"{load_checkpoint_path}", 
            "--trainer.wandb.resume", 
            "True", 
            "--trainer.id", 
            f"{trainer_id}"
        ], 
        capture_output=True, 
        text=True
    )

def start_inference(tpu_name, tpu_type, zone, config):
    return subprocess.run(
        [
            "sudo", 
            "python3", 
            "infra/launch.py", 
            "--tpu_name", 
            f"{tpu_name}", 
            "--tpu_type", 
            f"{tpu_type}", 
            "--zone", 
            f"{zone}", 
            "--", 
            "python3", 
            "src/levanter/main/train_lm.py", 
            "--config_path",
            f"{config}",
        ], 
        capture_output=True, 
        text=True
    )

def run_inference(tpu_name, tpu_type, zone, config, trainer_id : Optional[str] = None , load_checkpoint_path : Optional[str] = None):
    if trainer_id is not None and load_checkpoint_path is not None:
        return resume_inference(tpu_name, tpu_type, zone, config, trainer_id, load_checkpoint_path)
    else:
        return start_inference(tpu_name, tpu_type, zone, config)

def find_checkpoint_path(fs, checkpoint_path):
    model_folder = checkpoint_path.split('/')[-1]

    checkpoints = fs.ls(checkpoint_path)
    print(checkpoints)
    if len(checkpoints) == 0:
        logger.warning("No checkpoint found")
        return None, None
    elif len(checkpoints) > 1:
        logger.error("multiple checkpoints found")
        return None, None

    trainer_id = checkpoints[0].split('/')[-1]

    if trainer_id == model_folder:
        logger.warning("No checkpoints found")
        return None, None

    checkpoints_by_step = fs.ls(f'{checkpoint_path}/{trainer_id}')
    checkpoint_steps = [checkpoint.split('/')[-1] for checkpoint in checkpoints_by_step]
    steps = [int(step.split('-')[-1]) for step in checkpoint_steps]
    steps.sort(reverse=True)

    latest_step = steps[0]
    load_checkpoint_path = f'{checkpoint_path}/{trainer_id}/step-{latest_step}'
    return trainer_id, load_checkpoint_path

def check_container_status(tpu_name, zone):
    container_status = subprocess.run(
        [
            "sudo",
            "gcloud", 
            "compute",
            "tpus",
            "tpu-vm", 
            "ssh", 
            f"{tpu_name}", 
            f"--zone={zone}", 
            "--command='sudo docker ps'"
        ], 
        capture_output=True,
        text=True
    )

    if container_status.stdout != '':
        if 'levanter' in container_status.stdout:
            return Container_Status.RUNNING
        
    return Container_Status.NOTRUNNING


def parse_tpu_status(tpu_describe_query_output):
    # TODO: Add enum for tpu status

    if tpu_describe_query_output.stdout != '':
        query_output = tpu_describe_query_output.stdout
    else: 
        query_output = tpu_describe_query_output.stderr

    if Tpu_Status.READY in query_output:
        return Tpu_Status.READY
    elif Tpu_Status.CREATING in query_output:
        return Tpu_Status.CREATING
    elif Tpu_Status.PREEMPTING in query_output:
        return Tpu_Status.PREEMPTING
    elif Tpu_Status.DELETING in query_output:
        return Tpu_Status.DELETING
    elif Tpu_Status.PREEMPTED in query_output:
        return Tpu_Status.PREEMPTED
    elif Tpu_Status.DELETED in query_output:
        return Tpu_Status.DELETED
    elif Tpu_Status.NOTFOUND in query_output:
        return Tpu_Status.NOTFOUND
    
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)

    args = parser.parse_args()

    config_file = args.config_file
    checkpoint_path = None

    with open(config_file, 'r') as f:
        inference_args = json.load(f)

    print(inference_args)
    tpu_name = inference_args['tpu_name']
    tpu_type = inference_args['tpu_type']
    zone = inference_args['zone']
    config = inference_args['config']
    base_bucket_path = inference_args['base_bucket_path']
    if 'checkpoint_path' in inference_args.keys():
        checkpoint_path = inference_args['checkpoint_path']

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"{tpu_name}_inference.log", encoding='utf-8', level=logging.DEBUG)

    fs: AbstractFileSystem = fsspec.core.url_to_fs(base_bucket_path)[0]

    # print(checkpoint_path)
    if checkpoint_path is not None:     
        trainer_id, load_checkpoint_path = find_checkpoint_path(fs, checkpoint_path)
        if trainer_id is not None:
            resume_training = True
            logger.info(f'Last checkpoints found at load_checkpoint_path: {load_checkpoint_path}')

    # TODO: Add inference running status enum
    inference_running = False
    container_status = Container_Status.NOTRUNNING

    logger.info(f'Started Training at {datetime.datetime.now()}')
    logger.info(f'TIME: {datetime.datetime.now()}')

    while True:
        tpu_status = parse_tpu_status(describe_tpu_vm(tpu_name, zone))

        logger.info(f'TIME: {datetime.datetime.now()}')
        logger.info(f'TPU STATUS: {tpu_status}')
        
        if container_status == Container_Status.NOTRUNNING and tpu_status == Tpu_Status.READY: 
            container_status = check_container_status(tpu_name, zone)
            if container_status == Container_Status.RUNNING:
                inference_running = True

        if tpu_status == Tpu_Status.READY:
            if inference_running:
                logger.info(f'{datetime.datetime.now()} - TPU RUNNING AS WELL AS YOUR INFERENCE...')
                time.sleep(180)
                continue
            else:
                logger.info(f'{datetime.datetime.now()} - Resuming Inference')
                trainer_id, load_checkpoint_path = find_checkpoint_path(fs, checkpoint_path)
                if trainer_id is not None:
                    resume_training = True
                    logger.info(f'Last checkpoints found at load_checkpoint_path: {load_checkpoint_path}')
                inference_ssh = run_inference(tpu_name, tpu_type, zone, config, trainer_id, load_checkpoint_path)
                if inference_ssh.returncode == 0:
                    logger.info(f'{datetime.datetime.now()} - Inference Started Successfully')
                    inference_running = True
                else:
                    logger.error(f'{datetime.datetime.now()} - Error in inference: {inference_ssh.stderr}')                

        elif tpu_status == Tpu_Status.DELETING or tpu_status == Tpu_Status.PREEMPTING:
            inference_running = False
            container_status = Container_Status.NOTRUNNING
            logger.warning(f'{datetime.datetime.now()} - TPU STATUS: {tpu_status}')
            time.sleep(180)
        
        elif tpu_status == Tpu_Status.PREEMPTED:
            inference_running = False
            container_status = Container_Status.NOTRUNNING
            logger.info(f'{datetime.datetime.now()} - DELETING TPU')
            delete_request = delete_tpu_vm(tpu_name, zone)
            if delete_request.returncode == 0:
                logger.info(f'{datetime.datetime.now()} - Deleted Successfully {delete_request.stdout}')
            else:
                logger.error(f'{datetime.datetime.now()} - Error in deleting tpu {delete_request.stderr}')
            time.sleep(30)

        elif tpu_status == Tpu_Status.NOTFOUND or tpu_status == Tpu_Status.DELETED:
            # create_tpu_vm(tpu_name, tpu_type, zone)
            # time.sleep(30)
            # logger.info(f'{datetime.datetime.now()} - Resuming Inference')
            # inference_ssh = run_inference(tpu_name, tpu_type, zone, config, trainer_id, load_checkpoint_path)
            # if inference_ssh.returncode == 0:
            #     logger.info(f'{datetime.datetime.now()} - Inference Started Successfully {inference_ssh.stdout}')
            #     inference_running = True
            # else:
            #     logger.error(f'{datetime.datetime.now()} - Error in inference: {inference_ssh.stderr}')
            #     time.sleep(120)
            inference_running = False
            container_status = Container_Status.NOTRUNNING
            time.sleep(120)
        else:
            time.sleep(120)